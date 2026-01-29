import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from main import app
from database import Base, get_db
from config import settings

# Test database
TEST_DATABASE_URL = "sqlite:///./test_reviews.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def client():
    """Create test client with fresh database."""
    Base.metadata.create_all(bind=engine)
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def headers():
    """API headers with authentication."""
    return {"X-API-Key": settings.api_key}


@pytest.fixture
def sample_reviews():
    """Sample review data for testing."""
    return [
        {
            "id": "rev-001",
            "location": "Downtown",
            "rating": 5,
            "text": "Amazing service! The staff was incredibly friendly and helpful.",
            "date": "2026-01-15T10:00:00"
        },
        {
            "id": "rev-002",
            "location": "Uptown",
            "rating": 2,
            "text": "Terrible experience. The place was dirty and service was slow.",
            "date": "2026-01-16T14:30:00"
        },
        {
            "id": "rev-003",
            "location": "Downtown",
            "rating": 4,
            "text": "Good food but a bit expensive for what you get.",
            "date": "2026-01-17T12:00:00"
        }
    ]


# ===== Health Check Tests =====

def test_health_check(client):
    """Test health check endpoint (no auth required)."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs"] == "/docs"


# ===== Authentication Tests =====

def test_missing_api_key(client):
    """Test that endpoints require API key."""
    response = client.get("/api/reviews")
    assert response.status_code == 422  # Missing header


def test_invalid_api_key(client):
    """Test invalid API key rejection."""
    response = client.get("/api/reviews", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


# ===== Ingest Tests =====

def test_ingest_reviews_success(client, headers, sample_reviews):
    """Test successful review ingestion (happy path)."""
    response = client.post(
        "/api/ingest",
        json={"reviews": sample_reviews},
        headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] == 3
    assert "Successfully ingested" in data["message"]


def test_ingest_duplicate_reviews(client, headers, sample_reviews):
    """Test ingesting duplicate reviews (should update)."""
    # First ingestion
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    # Update review text
    sample_reviews[0]["text"] = "Updated review text"
    
    # Second ingestion
    response = client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    assert response.status_code == 200
    
    # Check review was updated
    review_response = client.get("/api/reviews/rev-001", headers=headers)
    assert review_response.status_code == 200
    assert "Updated review text" in review_response.json()["text"]


def test_ingest_invalid_rating(client, headers):
    """Test ingesting review with invalid rating (error path)."""
    invalid_review = {
        "reviews": [{
            "id": "rev-bad",
            "location": "Downtown",
            "rating": 6,  # Invalid: should be 1-5
            "text": "Test",
            "date": "2026-01-15T10:00:00"
        }]
    }
    response = client.post("/api/ingest", json=invalid_review, headers=headers)
    assert response.status_code == 422  # Validation error


# ===== Review Listing Tests =====

def test_get_reviews_empty(client, headers):
    """Test getting reviews when database is empty."""
    response = client.get("/api/reviews", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["reviews"] == []
    assert data["total_pages"] == 0


def test_get_reviews_with_data(client, headers, sample_reviews):
    """Test getting reviews after ingestion."""
    # Ingest reviews
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    # Get reviews
    response = client.get("/api/reviews", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["reviews"]) == 3


def test_get_reviews_filter_by_location(client, headers, sample_reviews):
    """Test filtering reviews by location."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/reviews?location=Downtown", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert all(r["location"] == "Downtown" for r in data["reviews"])


def test_get_reviews_filter_by_sentiment(client, headers, sample_reviews):
    """Test filtering reviews by sentiment."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/reviews?sentiment=positive", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert all(r["sentiment"] == "positive" for r in data["reviews"])


def test_get_reviews_text_search(client, headers, sample_reviews):
    """Test text search in reviews."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/reviews?q=food", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1


def test_get_reviews_pagination(client, headers, sample_reviews):
    """Test pagination."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    # Get first page with page_size=2
    response = client.get("/api/reviews?page=1&page_size=2", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["reviews"]) == 2
    assert data["page"] == 1
    assert data["total_pages"] == 2


# ===== Single Review Tests =====

def test_get_review_by_id(client, headers, sample_reviews):
    """Test getting a single review by ID."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/reviews/rev-001", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "rev-001"
    assert data["location"] == "Downtown"
    assert data["rating"] == 5


def test_get_review_not_found(client, headers):
    """Test getting non-existent review (error path)."""
    response = client.get("/api/reviews/nonexistent", headers=headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# ===== Suggest Reply Tests =====

def test_suggest_reply(client, headers, sample_reviews):
    """Test AI reply suggestion."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.post("/api/reviews/rev-001/suggest-reply", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "tags" in data
    assert "sentiment" in data["tags"]
    assert "topic" in data["tags"]
    assert "reasoning_log" in data
    assert len(data["reply"]) > 0


def test_suggest_reply_not_found(client, headers):
    """Test suggesting reply for non-existent review (error path)."""
    response = client.post("/api/reviews/nonexistent/suggest-reply", headers=headers)
    assert response.status_code == 404


# ===== Analytics Tests =====

def test_analytics_empty(client, headers):
    """Test analytics with no data."""
    response = client.get("/api/analytics", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 0
    assert data["avg_rating"] == 0.0
    assert data["sentiment_counts"] == {}
    assert data["topic_counts"] == {}


def test_analytics_with_data(client, headers, sample_reviews):
    """Test analytics with review data."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/analytics", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 3
    assert data["avg_rating"] > 0
    assert len(data["sentiment_counts"]) > 0
    assert len(data["topic_counts"]) > 0


# ===== Search Tests =====

def test_search_reviews(client, headers, sample_reviews):
    """Test TF-IDF search functionality."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/search?q=friendly staff service", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert data["query"] == "friendly staff service"


def test_search_missing_query(client, headers):
    """Test search without query parameter (error path)."""
    response = client.get("/api/search", headers=headers)
    assert response.status_code == 422  # Missing required parameter


def test_search_with_k_parameter(client, headers, sample_reviews):
    """Test search with custom k parameter."""
    client.post("/api/ingest", json={"reviews": sample_reviews}, headers=headers)
    
    response = client.get("/api/search?q=service&k=2", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) <= 2
