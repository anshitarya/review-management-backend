from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import math

from database import get_db, Review
from schemas import (
    IngestRequest, IngestResponse, ReviewResponse, ReviewListResponse,
    SuggestReplyResponse, AITags, AnalyticsResponse, SearchResponse,
    SearchResult, HealthResponse
)
from ai_service import ai_service
from search_service import search_service
from config import settings

router = APIRouter()


def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@router.post("/ingest", response_model=IngestResponse)
async def ingest_reviews(
    request: IngestRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Ingest a batch of reviews.
    
    Accepts an array of reviews and stores them in the database.
    Also analyzes sentiment and topic for each review.
    """
    try:
        count = 0
        for review_data in request.reviews:
            # Check if review already exists
            existing = db.query(Review).filter(Review.id == review_data.id).first()
            
            if existing:
                # Update existing review
                existing.location = review_data.location
                existing.rating = review_data.rating
                existing.text = review_data.text
                existing.date = review_data.date
                existing.updated_at = datetime.utcnow()
                review = existing
            else:
                # Create new review
                review = Review(
                    id=review_data.id,
                    location=review_data.location,
                    rating=review_data.rating,
                    text=review_data.text,
                    date=review_data.date
                )
                db.add(review)
            
            # Analyze sentiment and topic
            try:
                review.sentiment = ai_service.analyze_sentiment(review_data.text)
                review.topic = ai_service.extract_topic(review_data.text)
            except Exception as e:
                print(f"AI analysis error for review {review_data.id}: {e}")
                review.sentiment = 'neutral'
                review.topic = 'other'
            
            count += 1
        
        db.commit()
        
        # Refresh search index
        search_service.refresh_index(db)
        
        return IngestResponse(
            success=True,
            count=count,
            message=f"Successfully ingested {count} reviews"
        )
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/reviews", response_model=ReviewListResponse)
async def get_reviews(
    location: Optional[str] = Query(None, description="Filter by location"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    q: Optional[str] = Query(None, description="Text search query"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get reviews with filtering and pagination.
    
    Supports filtering by location, sentiment, and text search.
    """
    # Build query
    query = db.query(Review)
    
    # Apply filters
    if location:
        query = query.filter(Review.location == location)
    
    if sentiment:
        query = query.filter(Review.sentiment == sentiment)
    
    if q:
        # Simple text search
        query = query.filter(Review.text.contains(q))
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    reviews = query.order_by(Review.date.desc()).offset(offset).limit(page_size).all()
    
    # Calculate total pages
    total_pages = math.ceil(total / page_size) if total > 0 else 0
    
    return ReviewListResponse(
        reviews=[ReviewResponse.model_validate(r) for r in reviews],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.get("/reviews/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get a single review by ID."""
    review = db.query(Review).filter(Review.id == review_id).first()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return ReviewResponse.model_validate(review)


@router.post("/reviews/{review_id}/suggest-reply", response_model=SuggestReplyResponse)
async def suggest_reply(
    review_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Generate a suggested reply for a review using AI.
    
    Returns the suggested reply text, AI tags, and reasoning log.
    """
    review = db.query(Review).filter(Review.id == review_id).first()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    # Ensure sentiment and topic are set
    if not review.sentiment:
        review.sentiment = ai_service.analyze_sentiment(review.text)
        db.commit()
    
    if not review.topic:
        review.topic = ai_service.extract_topic(review.text)
        db.commit()
    
    # Generate reply
    try:
        reply_text, reasoning = ai_service.generate_reply(
            review.text,
            review.rating,
            review.sentiment,
            review.topic
        )
        
        return SuggestReplyResponse(
            reply=reply_text,
            tags=AITags(
                sentiment=review.sentiment,
                topic=review.topic
            ),
            reasoning_log=reasoning
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate reply: {str(e)}"
        )


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get analytics data for all reviews.
    
    Returns counts by sentiment and topic, plus overall statistics.
    """
    reviews = db.query(Review).all()
    
    if not reviews:
        return AnalyticsResponse(
            sentiment_counts={},
            topic_counts={},
            total_reviews=0,
            avg_rating=0.0
        )
    
    # Count by sentiment
    sentiment_counts = {}
    for review in reviews:
        sentiment = review.sentiment or 'unknown'
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    # Count by topic
    topic_counts = {}
    for review in reviews:
        topic = review.topic or 'unknown'
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Calculate average rating
    avg_rating = sum(r.rating for r in reviews) / len(reviews)
    
    return AnalyticsResponse(
        sentiment_counts=sentiment_counts,
        topic_counts=topic_counts,
        total_reviews=len(reviews),
        avg_rating=round(avg_rating, 2)
    )


@router.get("/search", response_model=SearchResponse)
async def search_reviews(
    q: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Search for similar reviews using TF-IDF and cosine similarity.
    
    Returns the top-k most similar reviews to the query.
    """
    # Perform search
    results = search_service.search(q, db, k)
    
    # Fetch review details
    search_results = []
    for review_id, score in results:
        review = db.query(Review).filter(Review.id == review_id).first()
        if review:
            search_results.append(
                SearchResult(
                    review=ReviewResponse.model_validate(review),
                    similarity_score=round(score, 4)
                )
            )
    
    return SearchResponse(
        results=search_results,
        query=q
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (no authentication required)."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )
