from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from database import Review
from sqlalchemy.orm import Session


class SearchService:
    """Service for TF-IDF based similarity search."""
    
    def __init__(self):
        """Initialize search service."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.review_vectors = None
        self.review_ids = []
    
    def index_reviews(self, db: Session):
        """
        Build TF-IDF index from all reviews in database.
        
        Args:
            db: Database session
        """
        reviews = db.query(Review).all()
        
        if not reviews:
            print("No reviews to index")
            return
        
        # Extract texts and IDs
        texts = [review.text for review in reviews]
        self.review_ids = [review.id for review in reviews]
        
        # Build TF-IDF vectors
        try:
            self.review_vectors = self.vectorizer.fit_transform(texts)
            print(f"✓ Indexed {len(reviews)} reviews for search")
        except Exception as e:
            print(f"Error indexing reviews: {e}")
            self.review_vectors = None
    
    def search(
        self,
        query: str,
        db: Session,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar reviews using cosine similarity.
        
        Args:
            query: Search query text
            db: Database session
            k: Number of results to return
            
        Returns:
            List of (review_id, similarity_score) tuples
        """
        # Re-index if needed
        if self.review_vectors is None:
            self.index_reviews(db)
        
        if self.review_vectors is None or len(self.review_ids) == 0:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.review_vectors)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Filter out results with very low similarity
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > 0.01:  # Minimum similarity threshold
                    results.append((self.review_ids[idx], float(score)))
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def refresh_index(self, db: Session):
        """Refresh the search index with latest reviews."""
        self.index_reviews(db)


# Global search service instance
search_service = SearchService()
