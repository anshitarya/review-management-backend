import re
from typing import Optional, Tuple
import cohere
from transformers import pipeline
from config import settings


class AIService:
    """Service for AI-powered features (sentiment analysis, reply generation)."""
    
    def __init__(self):
        """Initialize AI service with Cohere or local models."""
        self.use_local = settings.use_local_model or not settings.cohere_api_key
        
        if not self.use_local and settings.cohere_api_key:
            try:
                self.cohere_client = cohere.Client(settings.cohere_api_key)
                print("✓ Using Cohere API for AI features")
            except Exception as e:
                print(f"⚠ Cohere initialization failed: {e}. Falling back to local models.")
                self.use_local = True
        
        if self.use_local:
            print("✓ Using local transformers models for AI features")
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
            except Exception as e:
                print(f"⚠ Local model initialization failed: {e}")
                self.sentiment_analyzer = None
                self.summarizer = None
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text.
        
        Returns: 'positive', 'negative', or 'neutral'
        """
        if self.use_local and self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                
                # Map to our categories
                if label == 'positive' and score > 0.6:
                    return 'positive'
                elif label == 'negative' and score > 0.6:
                    return 'negative'
                else:
                    return 'neutral'
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
                return self._fallback_sentiment(text)
        else:
            # Use Cohere
            try:
                response = self.cohere_client.classify(
                    inputs=[text],
                    examples=[
                        cohere.ClassifyExample(text="Great service! Loved it!", label="positive"),
                        cohere.ClassifyExample(text="Amazing experience, will come back", label="positive"),
                        cohere.ClassifyExample(text="Terrible service, very disappointed", label="negative"),
                        cohere.ClassifyExample(text="Worst experience ever", label="negative"),
                        cohere.ClassifyExample(text="It was okay, nothing special", label="neutral"),
                    ]
                )
                return response.classifications[0].prediction
            except Exception as e:
                print(f"Cohere sentiment error: {e}")
                return self._fallback_sentiment(text)
    
    def extract_topic(self, text: str) -> str:
        """
        Extract main topic from review text.
        
        Returns: 'service', 'cleanliness', 'price', 'food', 'ambiance', or 'other'
        """
        text_lower = text.lower()
        
        # Simple keyword-based classification
        topics = {
            'service': ['service', 'staff', 'waiter', 'waitress', 'employee', 'friendly', 'rude', 'helpful', 'slow'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'mess', 'spotless', 'filthy'],
            'price': ['price', 'expensive', 'cheap', 'cost', 'value', 'overpriced', 'affordable', 'worth'],
            'food': ['food', 'meal', 'dish', 'taste', 'delicious', 'bland', 'fresh', 'stale', 'menu'],
            'ambiance': ['atmosphere', 'ambiance', 'decor', 'music', 'loud', 'quiet', 'cozy', 'comfortable']
        }
        
        topic_scores = {}
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return 'other'
    
    def generate_reply(
        self,
        review_text: str,
        rating: int,
        sentiment: str,
        topic: str
    ) -> Tuple[str, str]:
        """
        Generate a suggested reply for a review.
        
        Returns: (reply_text, reasoning_log)
        """
        # Sanitize the review text
        sanitized_text = self._sanitize_text(review_text)
        
        # Build context
        context = self._build_reply_context(rating, sentiment, topic)
        
        if not self.use_local and settings.cohere_api_key:
            try:
                return self._generate_with_cohere(sanitized_text, context, rating, sentiment)
            except Exception as e:
                print(f"Cohere generation error: {e}")
                return self._generate_with_local(sanitized_text, context, rating, sentiment)
        else:
            return self._generate_with_local(sanitized_text, context, rating, sentiment)
    
    def _generate_with_cohere(
        self,
        text: str,
        context: str,
        rating: int,
        sentiment: str
    ) -> Tuple[str, str]:
        """Generate reply using Cohere API."""
        prompt = f"""You are a customer service representative responding to a customer review.

Review (Rating: {rating}/5, Sentiment: {sentiment}):
{text}

{context}

Write a professional, empathetic response (2-3 sentences). Be concise and warm.

Response:"""
        
        response = self.cohere_client.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            stop_sequences=["\n\n"]
        )
        
        reply = response.generations[0].text.strip()
        reasoning = f"Generated using Cohere API with temperature=0.7, considering {sentiment} sentiment and {rating}-star rating."
        
        return reply, reasoning
    
    def _generate_with_local(
        self,
        text: str,
        context: str,
        rating: int,
        sentiment: str
    ) -> Tuple[str, str]:
        """Generate reply using local models or templates."""
        # Template-based responses
        templates = {
            'positive': [
                "Thank you so much for your wonderful feedback! We're thrilled to hear you had a great experience. We look forward to welcoming you back soon!",
                "We're so glad you enjoyed your visit! Your kind words mean a lot to our team. Hope to see you again!",
            ],
            'negative': [
                "We sincerely apologize for your disappointing experience. This is not the standard we aim for. Please contact us directly so we can make this right.",
                "Thank you for bringing this to our attention. We're truly sorry for falling short of your expectations and would like the opportunity to improve your experience.",
            ],
            'neutral': [
                "Thank you for taking the time to share your feedback. We appreciate your comments and are always working to improve. We hope to serve you better next time!",
            ]
        }
        
        # Select template based on sentiment
        import random
        reply = random.choice(templates.get(sentiment, templates['neutral']))
        
        reasoning = f"Generated using template-based approach for {sentiment} sentiment ({rating}-star rating)."
        
        return reply, reasoning
    
    def _build_reply_context(self, rating: int, sentiment: str, topic: str) -> str:
        """Build context string for reply generation."""
        context_parts = []
        
        if sentiment == 'negative':
            context_parts.append("Acknowledge the issue and apologize sincerely.")
        elif sentiment == 'positive':
            context_parts.append("Express gratitude and appreciation.")
        
        if topic == 'service':
            context_parts.append("Address service-related concerns.")
        elif topic == 'food':
            context_parts.append("Comment on the food quality feedback.")
        elif topic == 'cleanliness':
            context_parts.append("Take hygiene concerns seriously.")
        elif topic == 'price':
            context_parts.append("Acknowledge value concerns.")
        
        return " ".join(context_parts)
    
    def _sanitize_text(self, text: str) -> str:
        """Remove sensitive information from text."""
        # Redact email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Redact phone numbers (various formats)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        return text
    
    def _fallback_sentiment(self, text: str) -> str:
        """Fallback sentiment analysis using simple keyword matching."""
        text_lower = text.lower()
        
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'fantastic', 'awesome']
        negative_words = ['bad', 'terrible', 'worst', 'awful', 'horrible', 'poor', 'disappointing', 'hate']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'


# Global AI service instance
ai_service = AIService()
