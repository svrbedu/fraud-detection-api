import os
import pytest
from fastapi.testclient import TestClient
from main import app, Base, get_db, PredictionLog
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import AsyncMock, patch
from fastapi_limiter.depends import RateLimiter
from jose import jwt
from datetime import datetime, timedelta

# Fixture to ensure a dummy model exists for testing
@pytest.fixture(scope="module")
def setup_dummy_model():
    model_path = "fraud_model.joblib"
    created = False
    if not os.path.exists(model_path):
        # Create a dummy model if it doesn't exist
        X = pd.DataFrame({'amount': [10, 100], 'distance_from_home': [1, 50], 'hour_of_day': [12, 23]})
        y = np.array([0, 1])
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        joblib.dump(model, model_path)
        created = True
    
    yield
    
    # Optional: Cleanup if we created it
    if created and os.path.exists(model_path):
        os.remove(model_path)

# Setup in-memory database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def mock_rate_limiter():
    # Find the RateLimiter dependency in the app routes and override it
    # This avoids issues with patching class methods and signature inspection
    overrides = {}
    for route in app.routes:
        if hasattr(route, "dependencies"):
            for dep in route.dependencies:
                if isinstance(dep.dependency, RateLimiter):
                    async def no_limit():
                        return None
                    overrides[dep.dependency] = no_limit
    
    app.dependency_overrides.update(overrides)
    yield
    for key in overrides:
        app.dependency_overrides.pop(key, None)

@pytest.fixture
def client(setup_dummy_model, db_session, mock_rate_limiter):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    # Generate a valid token using the same secret/algo as main.py default
    expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode = {"sub": "testuser", "exp": expire}
    token = jwt.encode(to_encode, "supersecretkey", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_predict_legit(client, auth_headers):
    payload = {
        "amount": 20.0,
        "distance_from_home": 5.0,
        "hour_of_day": 14
    }
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert isinstance(data["is_fraud"], bool)

def test_predict_validation_error(client, auth_headers):
    payload = {
        "amount": -10, # Invalid amount
        "distance_from_home": 5.0,
        "hour_of_day": 25 # Invalid hour
    }
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 422

def test_predict_unauthorized(client):
    payload = {
        "amount": 20.0,
        "distance_from_home": 5.0,
        "hour_of_day": 14
    }
    # No headers provided
    response = client.post("/predict", json=payload)
    assert response.status_code == 401

def test_get_recent_alerts(client, db_session):
    # 1. Insert a fraud record directly into the test DB
    fraud_entry = PredictionLog(
        amount=1000.0,
        distance_from_home=100.0,
        hour_of_day=3,
        is_fraud=True,
        probability=0.99
    )
    db_session.add(fraud_entry)
    db_session.commit()

    # 2. Call the endpoint
    response = client.get("/alerts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["is_fraud"] is True
    assert data[0]["amount"] == 1000.0

def test_get_alerts_pagination(client, db_session):
    # 1. Insert 15 fraud records
    for i in range(15):
        entry = PredictionLog(
            amount=100.0 + i,
            distance_from_home=10.0,
            hour_of_day=12,
            is_fraud=True,
            probability=0.9
        )
        db_session.add(entry)
    db_session.commit()

    # 2. Get first page (limit 10)
    response = client.get("/alerts?skip=0&limit=10")
    assert len(response.json()) == 10

    # 3. Get second page (skip 10) - should have 5 remaining
    response = client.get("/alerts?skip=10&limit=10")
    assert len(response.json()) == 5

def test_get_fraud_stats(client, db_session):
    # 1. Insert fraud records for specific hours
    # 2 records at hour 10, 1 record at hour 15
    hours = [10, 10, 15]
    for h in hours:
        entry = PredictionLog(
            amount=500.0,
            distance_from_home=50.0,
            hour_of_day=h,
            is_fraud=True,
            probability=0.95
        )
        db_session.add(entry)
    db_session.commit()

    # 2. Call stats endpoint
    response = client.get("/stats/fraud-per-hour")
    assert response.status_code == 200
    data = response.json()
    
    # Convert list to dict for easier assertion: {hour: count}
    stats_map = {item['hour']: item['count'] for item in data}
    assert stats_map[10] == 2
    assert stats_map[15] == 1