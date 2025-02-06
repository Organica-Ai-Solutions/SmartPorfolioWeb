from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from database import SQLALCHEMY_DATABASE_URL
import os

def init_db():
    """Initialize the database with tables."""
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(SQLALCHEMY_DATABASE_URL.replace('sqlite:///', ''))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Create engine
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return engine, SessionLocal

if __name__ == "__main__":
    print("Initializing database...")
    engine, SessionLocal = init_db()
    print("Database initialized successfully!") 