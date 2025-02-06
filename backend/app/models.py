from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class RiskTolerance(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    portfolios = relationship("Portfolio", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    tickers = Column(JSON)  # List of tickers
    risk_tolerance = Column(String, default="medium")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="portfolios")
    snapshots = relationship("PortfolioSnapshot", back_populates="portfolio")
    rebalance_history = relationship("RebalanceHistory", back_populates="portfolio")

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    allocations = Column(JSON)  # Dictionary of ticker: weight
    metrics = Column(JSON)  # Portfolio metrics
    asset_metrics = Column(JSON)  # Individual asset metrics
    total_value = Column(Float)
    cash_position = Column(Float)
    
    portfolio = relationship("Portfolio", back_populates="snapshots")

class RebalanceHistory(Base):
    __tablename__ = "rebalance_history"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    old_allocations = Column(JSON)
    new_allocations = Column(JSON)
    orders = Column(JSON)  # List of executed orders
    status = Column(String)  # success, partial, failed
    
    portfolio = relationship("Portfolio", back_populates="rebalance_history")

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    additional_data = Column(JSON)  # For storing additional market data

class RiskAlert(Base):
    __tablename__ = "risk_alerts"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String)  # volatility, drawdown, concentration, etc.
    severity = Column(String)  # low, medium, high
    message = Column(String)
    metrics = Column(JSON)  # Relevant metrics that triggered the alert

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    type = Column(String)  # rebalance, risk_alert, performance, etc.
    title = Column(String)
    message = Column(String)
    read = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="notifications") 