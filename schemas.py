from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class ReviewBase(BaseModel):
    """Base review schema."""
    id: str
    location: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    text: str
    date: datetime


class ReviewCreate(ReviewBase):
    """Schema for creating a review."""
    pass


class ReviewInDB(ReviewBase):
    """Schema for review in database."""
    sentiment: Optional[str] = None
    topic: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ReviewResponse(ReviewInDB):
    """Schema for review API response."""
    pass


class IngestRequest(BaseModel):
    """Schema for bulk review ingestion."""
    reviews: List[ReviewCreate]


class IngestResponse(BaseModel):
    """Schema for ingestion response."""
    success: bool
    count: int
    message: str


class AITags(BaseModel):
    """AI-generated tags for a review."""
    sentiment: str
    topic: str


class SuggestReplyResponse(BaseModel):
    """Schema for suggested reply response."""
    reply: str
    tags: AITags
    reasoning_log: str


class ReviewListResponse(BaseModel):
    """Schema for paginated review list."""
    reviews: List[ReviewResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AnalyticsResponse(BaseModel):
    """Schema for analytics data."""
    sentiment_counts: dict
    topic_counts: dict
    total_reviews: int
    avg_rating: float


class SearchResult(BaseModel):
    """Schema for search result."""
    review: ReviewResponse
    similarity_score: float


class SearchResponse(BaseModel):
    """Schema for search response."""
    results: List[SearchResult]
    query: str


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    timestamp: datetime
    version: str
