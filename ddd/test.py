# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# PostgreSQL 연결 URL
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Engine 생성
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 연결 상태 확인
    echo=True  # SQL 쿼리 로깅 (개발 시)
)

# SessionLocal 클래스 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 생성
Base = declarative_base()


# Dependency: DB 세션 가져오기
def get_db():
    """
    FastAPI Dependency로 사용할 DB 세션
    요청마다 새로운 세션을 생성하고 종료 시 자동으로 닫음
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    데이터베이스 초기화
    모든 테이블 생성
    """
    # 모든 모델 import (테이블 생성 위해 필요)
    from app.models.photo import Event, PhotoLocation, Photo, User
    
    # 테이블 생성
    Base.metadata.create_all(bind=engine)
    print("✓ 데이터베이스 테이블이 생성되었습니다.")


def drop_all_tables():
    """
    모든 테이블 삭제 (주의: 개발/테스트 용도만)
    """
    Base.metadata.drop_all(bind=engine)
    print("✓ 모든 테이블이 삭제되었습니다.")