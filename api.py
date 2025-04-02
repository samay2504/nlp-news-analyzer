from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import logging
from utils import get_news_articles, analyze_sentiment, generate_summary, extract_topics, generate_tts, ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="News Analyzer API")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompanyRequest(BaseModel):
    company_name: str = Field(..., description="Name of the company to analyze")


@app.on_event("startup")
async def startup_event():
    """Initialize models when the API starts"""
    try:
        logger.info("Starting model initialization")
        model_manager = ModelManager()
        model_manager.initialize()
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/analyze")
async def analyze_news(request: CompanyRequest):
    """Analyze news articles for a given company"""
    company = request.company_name.strip()

    if not company:
        raise HTTPException(status_code=400, detail="Company name cannot be empty")

    # Get news articles
    logger.info(f"Fetching news for company: {company}")
    articles_data = get_news_articles(company)

    if not articles_data:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient articles found for {company}. Please try another company name."
        )

    # Process articles
    articles = []
    for art in articles_data:
        text = art['content']
        summary = generate_summary(text) if text else "No content available"
        sentiment = analyze_sentiment(text) if text else "Neutral"
        topics = extract_topics(text) if text else []

        articles.append({
            'title': art['title'],
            'summary': summary,
            'sentiment': sentiment,
            'topics': topics,
            'link': art['link']
        })

    # Calculate sentiment distribution
    sentiment_dist = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for art in articles:
        sentiment_dist[art['sentiment']] += 1

    # Collect all topics for analysis
    all_topics = []
    for art in articles:
        all_topics.extend(art['topics'])

    # Find common and unique topics
    topic_counts = {}
    for topic in all_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    common_topics = [t for t, count in topic_counts.items() if count > 1]
    unique_topics = [t for t, count in topic_counts.items() if count == 1]

    # Generate Hindi TTS summary using numerical values
    # (conversion to Hindi words happens inside generate_tts)
    total = len(articles)
    pos, neg, neu = sentiment_dist['Positive'], sentiment_dist['Negative'], sentiment_dist['Neutral']

    tts_text = (
        f"{company} के कुल {total} समाचार लेखों में "
        f"{pos} सकारात्मक, {neg} नकारात्मक और "
        f"{neu} तटस्थ लेख पाए गए।"
    )

    if common_topics:
        tts_text += " मुख्य विषय: " + ", ".join(common_topics[:3]) + "।"

    # Generate TTS audio
    tts_audio = generate_tts(tts_text)

    return {
        'company': company,
        'articles': articles,
        'sentiment_distribution': sentiment_dist,
        'common_topics': common_topics,
        'unique_topics': unique_topics,
        'comparative_analysis': {
            'sentiment_variation': f"Positive: {pos}, Negative: {neg}, Neutral: {neu}",
            'topic_insights': f"Common: {', '.join(common_topics) if common_topics else 'None'}"
        },
        'audio': tts_audio.hex(),
        'status': 'success'
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return {
        "status": "error",
        "detail": str(exc)
    }

# Run the API with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
