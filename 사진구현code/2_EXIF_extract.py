from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ExifExtractor:
    """EXIF 데이터 추출 및 파싱"""
    
    @staticmethod
    def extract_exif(image_path: str) -> Dict:
        """
        이미지에서 EXIF 데이터를 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            EXIF 데이터 딕셔너리
        """
        try:
            image = Image.open(image_path)
            exif_data = {}
            
            # EXIF 데이터 추출
            exif = image._getexif()
            
            if exif is None:
                logger.warning(f"No EXIF data found in {image_path}")
                return {}
            
            # 태그를 읽기 쉬운 이름으로 변환
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                
                # GPS 정보 별도 처리
                if tag_name == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id in value:
                        gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag_name] = value[gps_tag_id]
                    exif_data['GPSInfo'] = gps_data
                else:
                    # bytes 타입은 문자열로 변환
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    exif_data[tag_name] = value
            
            return exif_data
            
        except Exception as e:
            logger.error(f"Error extracting EXIF from {image_path}: {e}")
            return {}
    
    @staticmethod
    def get_datetime(exif_data: Dict) -> Optional[datetime]:
        """
        EXIF에서 촬영 시간 추출
        
        우선순위:
        1. DateTimeOriginal (원본 촬영 시간)
        2. DateTimeDigitized (디지털화 시간)
        3. DateTime (파일 수정 시간)
        """
        datetime_fields = ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']
        
        for field in datetime_fields:
            datetime_str = exif_data.get(field)
            if datetime_str:
                try:
                    # EXIF 날짜 형식: "2024:10:31 14:30:00"
                    return datetime.strptime(str(datetime_str), '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    logger.warning(f"Failed to parse datetime: {datetime_str}")
                    continue
        
        return None
    
    @staticmethod
    def get_gps_coordinates(exif_data: Dict) -> Optional[Tuple[float, float]]:
        """
        EXIF에서 GPS 좌표 추출
        
        Returns:
            (latitude, longitude) 또는 None
        """
        gps_info = exif_data.get('GPSInfo')
        if not gps_info:
            return None
        
        try:
            # GPS 좌표 변환
            lat = ExifExtractor._convert_to_degrees(
                gps_info.get('GPSLatitude'),
                gps_info.get('GPSLatitudeRef')
            )
            lon = ExifExtractor._convert_to_degrees(
                gps_info.get('GPSLongitude'),
                gps_info.get('GPSLongitudeRef')
            )
            
            if lat is not None and lon is not None:
                return (lat, lon)
                
        except Exception as e:
            logger.error(f"Error parsing GPS coordinates: {e}")
        
        return None
    
    @staticmethod
    def _convert_to_degrees(value, ref) -> Optional[float]:
        """GPS 좌표를 도(degree) 단위로 변환"""
        if not value or not ref:
            return None
        
        try:
            # value는 보통 ((도, 1), (분, 1), (초, 1)) 형식
            d = float(value[0][0]) / float(value[0][1])
            m = float(value[1][0]) / float(value[1][1])
            s = float(value[2][0]) / float(value[2][1])
            
            degrees = d + (m / 60.0) + (s / 3600.0)
            
            # 남위/서경은 음수로
            if ref in ['S', 'W']:
                degrees = -degrees
            
            return degrees
            
        except (IndexError, ZeroDivisionError, TypeError):
            return None
    
    @staticmethod
    def get_camera_info(exif_data: Dict) -> Dict[str, str]:
        """카메라 정보 추출"""
        return {
            'make': exif_data.get('Make', ''),
            'model': exif_data.get('Model', ''),
            'lens': exif_data.get('LensModel', ''),
        }
    
    @staticmethod
    def extract_all_info(image_path: str) -> Dict:
        """
        이미지에서 필요한 모든 정보 추출
        
        Returns:
            {
                'datetime': datetime 객체,
                'gps': (lat, lon) 또는 None,
                'camera': {...},
                'raw_exif': {...}
            }
        """
        exif_data = ExifExtractor.extract_exif(image_path)
        
        return {
            'datetime': ExifExtractor.get_datetime(exif_data),
            'gps': ExifExtractor.get_gps_coordinates(exif_data),
            'camera': ExifExtractor.get_camera_info(exif_data),
            'raw_exif': exif_data
        }