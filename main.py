from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db, get_db
from routes import router
from search_service import search_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    print("🚀 Starting Review Management API...")
    
    # Initialize database
    init_db()
    print("✓ Database initialized")
    
    # Build search index
    db = next(get_db())
    try:
        search_service.index_reviews(db)
    finally:
        db.close()
    
    yield
    
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Review Management API",
    description="Multi-location business review management system with AI-powered features",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://review-management-front.vercel.app",  # Your production frontend
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api", tags=["reviews"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Review Management API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    from config import settings
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
