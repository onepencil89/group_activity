from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PhotoClassifier:
    """시간 기반 사진 자동 분류"""
    
    def __init__(self, time_gap_threshold: int = 1800):
        """
        Args:
            time_gap_threshold: 다른 위치로 간주할 시간 간격 (초), 기본 30분
        """
        self.time_gap_threshold = time_gap_threshold
    
    def analyze_photos(self, photos: List[Dict]) -> Dict:
        """
        사진들을 분석하여 촬영 패턴 파악
        
        Args:
            photos: [{'id': 1, 'filename': 'IMG001.jpg', 'datetime': datetime_obj}, ...]
            
        Returns:
            {
                'total_count': 전체 사진 수,
                'time_range': (시작 시간, 종료 시간),
                'suggested_groups': 제안된 그룹 수,
                'groups': 그룹 정보 리스트
            }
        """
        if not photos:
            return {'total_count': 0, 'groups': []}
        
        # 시간순 정렬
        sorted_photos = sorted(photos, key=lambda x: x['datetime'])
        
        # 시간 간격 분석
        time_gaps = self._analyze_time_gaps(sorted_photos)
        
        # 그룹으로 분할
        groups = self._split_into_groups(sorted_photos, time_gaps)
        
        return {
            'total_count': len(photos),
            'time_range': (
                sorted_photos[0]['datetime'],
                sorted_photos[-1]['datetime']
            ),
            'suggested_groups': len(groups),
            'groups': groups,
            'time_gaps': time_gaps
        }
    
    def _analyze_time_gaps(self, sorted_photos: List[Dict]) -> List[Dict]:
        """연속된 사진들 간의 시간 간격 분석"""
        gaps = []
        
        for i in range(len(sorted_photos) - 1):
            current = sorted_photos[i]
            next_photo = sorted_photos[i + 1]
            
            gap_seconds = (next_photo['datetime'] - current['datetime']).total_seconds()
            
            gaps.append({
                'index': i,
                'gap_seconds': gap_seconds,
                'from': current['datetime'],
                'to': next_photo['datetime'],
                'is_significant': gap_seconds > self.time_gap_threshold
            })
        
        return gaps
    
    def _split_into_groups(self, sorted_photos: List[Dict], time_gaps: List[Dict]) -> List[Dict]:
        """시간 간격이 큰 지점을 기준으로 그룹 분할"""
        groups = []
        current_group = []
        
        for i, photo in enumerate(sorted_photos):
            current_group.append(photo)
            
            # 다음 사진과의 간격이 크면 그룹 종료
            if i < len(time_gaps) and time_gaps[i]['is_significant']:
                groups.append(self._create_group_info(current_group, len(groups) + 1))
                current_group = []
        
        # 마지막 그룹 추가
        if current_group:
            groups.append(self._create_group_info(current_group, len(groups) + 1))
        
        return groups
    
    def _create_group_info(self, photos: List[Dict], group_number: int) -> Dict:
        """그룹 정보 생성"""
        return {
            'group_number': group_number,
            'photo_count': len(photos),
            'time_range_start': photos[0]['datetime'],
            'time_range_end': photos[-1]['datetime'],
            'duration_minutes': (photos[-1]['datetime'] - photos[0]['datetime']).total_seconds() / 60,
            'photo_ids': [p['id'] for p in photos],
            'suggested_name': f"촬영지점 {group_number}",
            # 첫 번째 사진을 대표 이미지로
            'representative_photo': photos[0]
        }
    
    def assign_photos_to_locations(
        self, 
        photos: List[Dict], 
        locations: List[Dict]
    ) -> List[Dict]:
        """
        사진들을 지정된 위치에 자동 할당
        
        Args:
            photos: 사진 리스트 (시간순 정렬됨)
            locations: 작가가 지정한 위치 리스트
                       [{'name': '5km지점', 'coords': (lat, lng)}, ...]
        
        Returns:
            위치별로 사진이 할당된 리스트
        """
        # 먼저 그룹 분석
        analysis = self.analyze_photos(photos)
        groups = analysis['groups']
        
        if len(groups) != len(locations):
            logger.warning(
                f"Group count ({len(groups)}) doesn't match location count ({len(locations)})"
            )
        
        # 그룹과 위치를 1:1 매칭
        for i, group in enumerate(groups):
            if i < len(locations):
                locations[i]['photos'] = [
                    p for p in photos if p['id'] in group['photo_ids']
                ]
                locations[i]['time_range_start'] = group['time_range_start']
                locations[i]['time_range_end'] = group['time_range_end']
                locations[i]['photo_count'] = group['photo_count']
                locations[i]['duration_minutes'] = group['duration_minutes']
        
        return locations
    
    def get_timeline_data(self, photos: List[Dict], interval_minutes: int = 30) -> List[Dict]:
        """
        타임라인 시각화를 위한 데이터 생성
        
        Args:
            photos: 사진 리스트
            interval_minutes: 시간 간격 (분)
            
        Returns:
            시간대별 사진 수 리스트
        """
        if not photos:
            return []
        
        sorted_photos = sorted(photos, key=lambda x: x['datetime'])
        start_time = sorted_photos[0]['datetime']
        end_time = sorted_photos[-1]['datetime']
        
        # 시간대별 버킷 생성
        timeline = []
        current_time = start_time
        interval = timedelta(minutes=interval_minutes)
        
        while current_time <= end_time:
            next_time = current_time + interval
            
            # 현재 시간대의 사진들 카운트
            photos_in_range = [
                p for p in sorted_photos 
                if current_time <= p['datetime'] < next_time
            ]
            
            timeline.append({
                'time_start': current_time,
                'time_end': next_time,
                'photo_count': len(photos_in_range),
                'photos': photos_in_range[:5]  # 최대 5개만 미리보기
            })
            
            current_time = next_time
        
        return timeline
    
    def suggest_location_names(self, groups: List[Dict], event_info: Dict = None) -> List[str]:
        """
        촬영 그룹에 대한 위치명 자동 제안
        
        Args:
            groups: 그룹 정보 리스트
            event_info: 대회 정보 (코스, 거리 등)
            
        Returns:
            제안된 위치명 리스트
        """
        suggestions = []
        
        for i, group in enumerate(groups):
            # 기본 제안
            suggestion = f"촬영 지점 {i + 1}"
            
            # 시간 기반 추가 정보
            time = group['time_range_start']
            suggestion += f" ({time.strftime('%H:%M')})"
            
            suggestions.append(suggestion)
        
        return suggestions