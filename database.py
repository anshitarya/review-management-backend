from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class Review(Base):
    """Review database model."""
    __tablename__ = "reviews"
    
    id = Column(String, primary_key=True, index=True)
    location = Column(String, index=True, nullable=False)
    rating = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    date = Column(DateTime, nullable=False)
    
    # AI-generated fields
    sentiment = Column(String, index=True, nullable=True)
    topic = Column(String, index=True, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Dependency to get DB session
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
