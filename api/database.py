from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/churn_db')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(100), index=True)
    churn_probability = Column(Float)
    prediction = Column(Integer)
    risk_level = Column(String(20))
    recommendation = Column(String(500))
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables in DB"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created")


def get_db():
    """Get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db, customer_id: str, result: dict):
    ""Save prediction to DB"""
    record = Prediction(
        customer_id=customer_id,
        churn_probability=result['churn_probability'],
        prediction=result['prediction'],
        risk_level=result['risk_level'],
        recommendation=result['recommendation'],
        model_version=result['model_version']
    )
    db.add(record)
    db.commit()
    return record
