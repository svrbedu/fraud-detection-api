import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Define Input Schema
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the transaction (0-23)")

# Define Output Schema
class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float

class AlertResponse(BaseModel):
    id: int
    timestamp: datetime
    amount: float
    distance_from_home: float
    hour_of_day: int
    is_fraud: bool
    probability: float

    class Config:
        from_attributes = True

class FraudStats(BaseModel):
    hour: int
    count: int

class Token(BaseModel):
    access_token: str
    token_type: str

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# Database Setup
# Using SQLite for local development ease, but compatible with Postgres via connection string
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_logs.db")

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float)
    distance_from_home = Column(Float)
    hour_of_day = Column(Integer)
    is_fraud = Column(Boolean)
    probability = Column(Float)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global model variable
model = None
MODEL_PATH = "fraud_model.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Init Redis Rate Limiter
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        redis_connection = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(redis_connection)
    except Exception as e:
        print(f"Warning: Could not connect to Redis for rate limiting: {e}")

    # Load model on startup
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Run train_model.py first.")
    yield
    # Clean up on shutdown
    model = None

app = FastAPI(title="Fraud Detection Microservice", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. Visit /docs for documentation."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real app, verify username/password from DB here.
    # For this example, we accept any user and issue a token.
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60))])
def predict(transaction: Transaction, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare input dataframe matching training features
    input_data = pd.DataFrame([transaction.model_dump()])
    
    # Predict
    try:
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = model.predict_proba(input_data)[0]
        fraud_prob = float(probs[1])
        is_fraud = fraud_prob > 0.5
        
        # Log request and prediction to database
        log_entry = PredictionLog(
            amount=transaction.amount,
            distance_from_home=transaction.distance_from_home,
            hour_of_day=transaction.hour_of_day,
            is_fraud=is_fraud,
            probability=fraud_prob
        )
        db.add(log_entry)
        db.commit()

        return PredictionResponse(is_fraud=is_fraud, probability=fraud_prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts", response_model=List[AlertResponse])
def get_recent_alerts(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    alerts = db.query(PredictionLog)\
        .filter(PredictionLog.is_fraud == True)\
        .order_by(PredictionLog.timestamp.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    return alerts

@app.get("/stats/fraud-per-hour", response_model=List[FraudStats])
def get_fraud_stats(db: Session = Depends(get_db)):
    stats = db.query(
        PredictionLog.hour_of_day,
        func.count(PredictionLog.id)
    ).filter(
        PredictionLog.is_fraud == True
    ).group_by(
        PredictionLog.hour_of_day
    ).all()
    
    return [{"hour": hour, "count": count} for hour, count in stats]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)