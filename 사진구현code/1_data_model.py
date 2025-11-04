from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON, BigInteger
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime
from app.database import Base


class Event(Base):
    """마라톤 대회 정보"""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    event_date = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=False)
    course_data = Column(JSON)  # 코스 GPS 경로
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    locations = relationship("PhotoLocation", back_populates="event")


class PhotoLocation(Base):
    """촬영 위치 정보"""
    __tablename__ = "photo_locations"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    photographer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    location_name = Column(String(100))  # "5km 지점", "결승점" 등
    coordinates = Column(Geometry('POINT'))  # PostGIS
    distance_from_start = Column(Float)  # km
    
    time_range_start = Column(DateTime)
    time_range_end = Column(DateTime)
    photo_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("Event", back_populates="locations")
    photos = relationship("Photo", back_populates="location")
    photographer = relationship("User", back_populates="photo_locations")


class Photo(Base):
    """사진 정보"""
    __tablename__ = "photos"
    
    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(Integer, ForeignKey("photo_locations.id"), nullable=True)
    
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500), nullable=False)
    thumbnail_path = Column(String(500))
    watermarked_path = Column(String(500))
    
    taken_at = Column(DateTime, nullable=False, index=True)
    file_size = Column(BigInteger)
    
    upload_batch_id = Column(String(50), index=True)  # 같이 업로드된 그룹
    exif_data = Column(JSON)  # 전체 EXIF 데이터 저장
    
    is_processed = Column(Integer, default=0)  # 0: 대기, 1: 처리중, 2: 완료
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    location = relationship("PhotoLocation", back_populates="photos")


class User(Base):
    """사용자 (작가/참가자)"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(100))
    user_type = Column(String(20))  # 'photographer' or 'runner'
    
    # 작가 전용
    watermark_config = Column(JSON)  # 워터마크 설정 저장
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    photo_locations = relationship("PhotoLocation", back_populates="photographer")


# 인덱스 생성 (성능 최적화)
from sqlalchemy import Index

Index('idx_photos_taken_at', Photo.taken_at)
Index('idx_photos_location', Photo.location_id)
Index('idx_photos_batch', Photo.upload_batch_id)