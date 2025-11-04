from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from typing import List
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime
import shutil

from app.database import get_db
from app.models.photo import Photo, PhotoLocation
from app.services.exif_extractor import ExifExtractor
from app.services.photo_classifier import PhotoClassifier
from app.tasks.photo_processing import process_photo_batch

router = APIRouter()


@router.post("/upload/batch")
async def upload_photos_batch(
    event_id: int,
    location_id: int = None,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    여러 사진을 한번에 업로드
    
    Args:
        event_id: 대회 ID
        location_id: 촬영 위치 ID (선택사항, 나중에 지정 가능)
        files: 업로드할 파일들
    """
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="최대 1000장까지 업로드 가능합니다")
    
    # 업로드 배치 ID 생성
    batch_id = str(uuid.uuid4())
    upload_dir = f"uploads/{batch_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded_photos = []
    exif_extractor = ExifExtractor()
    
    # 각 파일 처리
    for file in files:
        # 파일 저장
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # EXIF 추출
        try:
            exif_info = exif_extractor.extract_all_info(file_path)
            
            # 촬영 시간이 없으면 파일 생성 시간 사용
            taken_at = exif_info['datetime']
            if not taken_at:
                taken_at = datetime.fromtimestamp(os.path.getctime(file_path))
            
            # DB에 저장
            photo = Photo(
                location_id=location_id,
                filename=file.filename,
                original_path=file_path,
                taken_at=taken_at,
                file_size=os.path.getsize(file_path),
                upload_batch_id=batch_id,
                exif_data=exif_info['raw_exif'],
                is_processed=0
            )
            
            db.add(photo)
            uploaded_photos.append({
                'filename': file.filename,
                'taken_at': taken_at,
                'has_gps': exif_info['gps'] is not None
            })
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue
    
    db.commit()
    
    # 백그라운드에서 썸네일 생성 등 추가 처리
    if background_tasks:
        background_tasks.add_task(process_photo_batch, batch_id)
    
    return {
        'batch_id': batch_id,
        'uploaded_count': len(uploaded_photos),
        'photos': uploaded_photos
    }


@router.post("/upload/analyze")
async def analyze_upload_batch(
    batch_id: str,
    db: Session = Depends(get_db)
):
    """
    업로드된 사진 배치를 분석하여 자동 분류 제안
    
    Returns:
        - 전체 사진 수
        - 시간 범위
        - 제안된 그룹 수
        - 그룹별 상세 정보
    """
    # 배치의 모든 사진 조회
    photos = db.query(Photo).filter(Photo.upload_batch_id == batch_id).all()
    
    if not photos:
        raise HTTPException(status_code=404, detail="사진을 찾을 수 없습니다")
    
    # 분류기로 분석
    classifier = PhotoClassifier(time_gap_threshold=1800)  # 30분
    
    photo_data = [
        {
            'id': p.id,
            'filename': p.filename,
            'datetime': p.taken_at
        }
        for p in photos
    ]
    
    analysis = classifier.analyze_photos(photo_data)
    
    # 타임라인 데이터도 생성
    timeline = classifier.get_timeline_data(photo_data, interval_minutes=30)
    
    return {
        'batch_id': batch_id,
        'analysis': analysis,
        'timeline': timeline,
        'suggestion': {
            'message': f"{analysis['suggested_groups']}개의 촬영 지점이 감지되었습니다.",
            'action': '지도에서 각 지점의 위치를 지정해주세요.'
        }
    }


@router.post("/upload/assign-locations")
async def assign_photos_to_locations(
    batch_id: str,
    locations: List[dict],
    db: Session = Depends(get_db)
):
    """
    작가가 지정한 위치에 사진들을 자동 할당
    
    Args:
        batch_id: 업로드 배치 ID
        locations: [
            {'name': '5km지점', 'lat': 37.5, 'lng': 127.0, 'distance_km': 5},
            ...
        ]
    """
    photos = db.query(Photo).filter(Photo.upload_batch_id == batch_id).all()
    
    if not photos:
        raise HTTPException(status_code=404, detail="사진을 찾을 수 없습니다")
    
    # 위치 정보 DB에 저장
    photo_locations = []
    for loc in locations:
        photo_loc = PhotoLocation(
            event_id=loc.get('event_id'),
            photographer_id=loc.get('photographer_id'),
            location_name=loc['name'],
            coordinates=f"POINT({loc['lng']} {loc['lat']})",
            distance_from_start=loc.get('distance_km')
        )
        db.add(photo_loc)
        db.flush()  # ID 생성
        photo_locations.append(photo_loc)
    
    # 사진을 위치에 자동 할당
    classifier = PhotoClassifier()
    
    photo_data = [
        {
            'id': p.id,
            'filename': p.filename,
            'datetime': p.taken_at
        }
        for p in photos
    ]
    
    location_data = [
        {
            'id': loc.id,
            'name': loc.location_name
        }
        for loc in photo_locations
    ]
    
    assigned = classifier.assign_photos_to_locations(photo_data, location_data)
    
    # DB 업데이트
    for i, loc_info in enumerate(assigned):
        location = photo_locations[i]
        location.time_range_start = loc_info['time_range_start']
        location.time_range_end = loc_info['time_range_end']
        location.photo_count = loc_info['photo_count']
        
        # 사진들의 location_id 업데이트
        for photo_id in loc_info.get('photo_ids', []):
            photo = db.query(Photo).filter(Photo.id == photo_id).first()
            if photo:
                photo.location_id = location.id
    
    db.commit()
    
    return {
        'message': '사진이 위치에 할당되었습니다',
        'locations': [
            {
                'id': loc.id,
                'name': loc.location_name,
                'photo_count': loc.photo_count,
                'time_range': f"{loc.time_range_start} ~ {loc.time_range_end}"
            }
            for loc in photo_locations
        ]
    }


@router.get("/batch/{batch_id}/timeline")
async def get_batch_timeline(
    batch_id: str,
    interval_minutes: int = 30,
    db: Session = Depends(get_db)
):
    """
    배치의 타임라인 데이터 조회 (시각화용)
    
    Args:
        batch_id: 배치 ID
        interval_minutes: 시간 간격 (분)
    """
    photos = db.query(Photo).filter(Photo.upload_batch_id == batch_id).all()
    
    if not photos:
        raise HTTPException(status_code=404, detail="사진을 찾을 수 없습니다")
    
    classifier = PhotoClassifier()
    photo_data = [
        {
            'id': p.id,
            'filename': p.filename,
            'datetime': p.taken_at,
            'thumbnail': p.thumbnail_path
        }
        for p in photos
    ]
    
    timeline = classifier.get_timeline_data(photo_data, interval_minutes)
    
    return {
        'batch_id': batch_id,
        'total_photos': len(photos),
        'timeline': timeline
    }