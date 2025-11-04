from celery import Celery
from PIL import Image
import os
from typing import List
from app.models.photo import Photo
from app.database import SessionLocal

# Celery 앱 설정
celery_app = Celery(
    'photo_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)


@celery_app.task(name='process_photo_batch')
def process_photo_batch(batch_id: str):
    """
    사진 배치를 백그라운드에서 처리
    - 썸네일 생성
    - 이미지 최적화
    - 메타데이터 추가 추출
    """
    db = SessionLocal()
    
    try:
        photos = db.query(Photo).filter(
            Photo.upload_batch_id == batch_id,
            Photo.is_processed == 0
        ).all()
        
        total = len(photos)
        processed = 0
        
        for photo in photos:
            try:
                # 썸네일 생성
                thumbnail_path = create_thumbnail(photo.original_path)
                photo.thumbnail_path = thumbnail_path
                
                # 처리 완료 표시
                photo.is_processed = 2
                processed += 1
                
                # 진행률 업데이트 (WebSocket으로 전송 가능)
                progress = (processed / total) * 100
                update_progress.delay(batch_id, progress)
                
            except Exception as e:
                print(f"Error processing photo {photo.id}: {e}")
                photo.is_processed = -1  # 실패 표시
        
        db.commit()
        
        return {
            'batch_id': batch_id,
            'total': total,
            'processed': processed
        }
        
    finally:
        db.close()


@celery_app.task(name='create_thumbnail')
def create_thumbnail(image_path: str, size: tuple = (400, 400)) -> str:
    """
    썸네일 생성
    
    Args:
        image_path: 원본 이미지 경로
        size: 썸네일 크기 (width, height)
    
    Returns:
        썸네일 파일 경로
    """
    try:
        # 썸네일 디렉토리 생성
        thumbnail_dir = os.path.join(os.path.dirname(image_path), 'thumbnails')
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # 썸네일 경로
        filename = os.path.basename(image_path)
        thumbnail_path = os.path.join(thumbnail_dir, f"thumb_{filename}")
        
        # 이미지 열기 및 리사이즈
        with Image.open(image_path) as img:
            # EXIF 방향 정보 적용
            img = apply_exif_orientation(img)
            
            # 비율 유지하며 리사이즈
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # 저장 (품질 85, 최적화)
            img.save(thumbnail_path, quality=85, optimize=True)
        
        return thumbnail_path
        
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None


def apply_exif_orientation(img: Image.Image) -> Image.Image:
    """
    EXIF 방향 정보에 따라 이미지 회전
    """
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(274)  # 274 = Orientation tag
            
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass
    
    return img


@celery_app.task(name='generate_watermarked_version')
def generate_watermarked_version(photo_id: int, watermark_config: dict):
    """
    워터마크가 적용된 버전 생성
    
    Args:
        photo_id: 사진 ID
        watermark_config: {
            'text': '© 2024 MyPhotography',
            'position': 'bottom-right',
            'opacity': 0.5,
            'font_size': 24
        }
    """
    db = SessionLocal()
    
    try:
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            return None
        
        # 워터마크 디렉토리
        watermark_dir = os.path.join(os.path.dirname(photo.original_path), 'watermarked')
        os.makedirs(watermark_dir, exist_ok=True)
        
        filename = os.path.basename(photo.original_path)
        watermarked_path = os.path.join(watermark_dir, f"wm_{filename}")
        
        # 워터마크 적용
        add_watermark(
            photo.original_path,
            watermarked_path,
            watermark_config
        )
        
        photo.watermarked_path = watermarked_path
        db.commit()
        
        return watermarked_path
        
    finally:
        db.close()


def add_watermark(input_path: str, output_path: str, config: dict):
    """
    이미지에 워터마크 추가
    """
    from PIL import ImageDraw, ImageFont
    
    with Image.open(input_path) as img:
        # 투명 레이어 생성
        watermark_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark_layer)
        
        # 폰트 설정
        try:
            font = ImageFont.truetype("arial.ttf", config.get('font_size', 24))
        except:
            font = ImageFont.load_default()
        
        # 텍스트 크기 계산
        text = config.get('text', '© Photography')
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 위치 계산
        position = config.get('position', 'bottom-right')
        padding = 20
        
        if position == 'bottom-right':
            x = img.width - text_width - padding
            y = img.height - text_height - padding
        elif position == 'bottom-left':
            x = padding
            y = img.height - text_height - padding
        elif position == 'top-right':
            x = img.width - text_width - padding
            y = padding
        else:  # top-left
            x = padding
            y = padding
        
        # 투명도 적용
        opacity = int(config.get('opacity', 0.5) * 255)
        
        # 텍스트 그리기
        draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)
        
        # 원본 이미지와 합성
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        watermarked = Image.alpha_composite(img, watermark_layer)
        
        # RGB로 변환 후 저장
        watermarked = watermarked.convert('RGB')
        watermarked.save(output_path, quality=95)


@celery_app.task(name='update_progress')
def update_progress(batch_id: str, progress: float):
    """
    처리 진행률 업데이트 (WebSocket이나 Redis에 저장)
    """
    # Redis에 진행률 저장
    from redis import Redis
    r = Redis(host='localhost', port=6379, db=0)
    r.setex(f"progress:{batch_id}", 3600, str(progress))  # 1시간 TTL
    
    return {'batch_id': batch_id, 'progress': progress}