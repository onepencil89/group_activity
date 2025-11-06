# app/models/photo.py (MySQL MVP 버전)
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON, BigInteger, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    """사용자 (작가만 MVP에서 사용)"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(100))
    password_hash = Column(String(255))  # 실제로는 해싱 필요
    
    # 작가 전용 설정
    watermark_text = Column(String(200))
    watermark_position = Column(String(20), default='bottom-right')  # bottom-right, bottom-left, etc
    watermark_opacity = Column(Float, default=0.5)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    events = relationship("Event", back_populates="photographer")


class Event(Base):
    """마라톤 대회"""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    photographer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    event_date = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    photographer = relationship("User", back_populates="events")
    locations = relationship("PhotoLocation", back_populates="event", cascade="all, delete-orphan")


class PhotoLocation(Base):
    """촬영 위치"""
    __tablename__ = "photo_locations"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    
    location_name = Column(String(100))  # "5km 지점"
    distance_km = Column(Float)  # 거리 (km)
    
    # GPS 좌표 (MySQL은 POINT 타입 대신 별도 컬럼 사용)
    latitude = Column(Float)
    longitude = Column(Float)
    
    # 촬영 시간 범위
    time_range_start = Column(DateTime, index=True)
    time_range_end = Column(DateTime, index=True)
    photo_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    event = relationship("Event", back_populates="locations")
    photos = relationship("Photo", back_populates="location", cascade="all, delete-orphan")


class Photo(Base):
    """사진"""
    __tablename__ = "photos"
    
    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(Integer, ForeignKey("photo_locations.id"), nullable=True)
    
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500), nullable=False)
    thumbnail_path = Column(String(500))
    watermarked_path = Column(String(500))
    
    taken_at = Column(DateTime, nullable=False, index=True)
    file_size = Column(BigInteger)
    
    upload_batch_id = Column(String(50), index=True)
    
    # 처리 상태: 0=대기, 1=처리중, 2=완료
    is_processed = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    location = relationship("PhotoLocation", back_populates="photos")


# 인덱스 (성능 최적화)
from sqlalchemy import Index

Index('idx_photos_taken_at', Photo.taken_at)
Index('idx_photos_location', Photo.location_id)
Index('idx_photos_batch', Photo.upload_batch_id)
Index('idx_location_time_start', PhotoLocation.time_range_start)
Index('idx_location_time_end', PhotoLocation.time_range_end)