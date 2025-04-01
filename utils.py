import feedparser
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
from transformers import pipeline, VitsModel, AutoTokenizer
import io
import soundfile as sf
import torch
import logging
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Singleton class for model management
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        """Initialize all models only once"""
        if self.initialized:
            return

        logger.info("Initializing NLP models...")
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.kw_model = KeyBERT()

            # Summarization model
            try:
                self.summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
            except Exception as e:
                logger.error(f"Failed to load summarizer: {str(e)}")
                self.summarizer = None

            # TTS model for Hindi
            try:
                self.tts_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
                self.tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
            except Exception as e:
                logger.error(f"Failed to load TTS model: {str(e)}")
                self.tts_model = None
                self.tts_tokenizer = None

            self.initialized = True
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise


def get_news_articles(company: str) -> List[Dict[str, Any]]:
    """Scrape news articles using Google News RSS with better error handling."""
    url = f'https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en'

    try:
        feed = feedparser.parse(url)
        articles = []

        if not feed.entries:
            logger.warning(f"No news found for company: {company}")
            return []

        for entry in feed.entries[:10]:  # Limit to first 10 articles
            article_data = {
                'title': entry.title,
                'link': entry.link,
                'content': ""
            }

            # Try to get article content
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(entry.link, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract paragraphs
                paragraphs = soup.find_all('p')
                content_text = []

                for p in paragraphs[:10]:  # First 10 paragraphs
                    text = p.get_text(strip=True)
                    # Skip short paragraphs and known filler
                    if len(text) < 20 or "submit your" in text.lower() or "copyright" in text.lower():
                        continue
                    content_text.append(text)

                # If we found content, use it
                if content_text:
                    article_data['content'] = " ".join(content_text)
                # Otherwise, fall back to the entry summary if available
                elif hasattr(entry, 'summary'):
                    article_data['content'] = entry.summary

                # Ensure minimum content length
                if len(article_data['content'].split()) >= 20:
                    articles.append(article_data)

            except Exception as e:
                logger.warning(f"Failed to fetch content for {entry.link}: {str(e)}")
                # Still add the article with just title and link if fetching content failed
                if hasattr(entry, 'summary') and entry.summary:
                    article_data['content'] = entry.summary
                    articles.append(article_data)

        logger.info(f"Found {len(articles)} valid articles for {company}")
        return articles

    except Exception as e:
        logger.error(f"Error fetching news for {company}: {str(e)}")
        return []


def analyze_sentiment(text: str) -> str:
    """Analyze sentiment using VADER."""
    if not text:
        return "Neutral"

    try:
        model_manager = ModelManager()
        if not model_manager.initialized:
            model_manager.initialize()

        scores = model_manager.sia.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'Positive'
        elif scores['compound'] <= -0.05:
            return 'Negative'
        return 'Neutral'
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return 'Neutral'


def generate_summary(text: str) -> str:
    """Generate a summary of the text."""
    if not text or len(text.split()) < 30:
        return text

    try:
        model_manager = ModelManager()
        if not model_manager.initialized:
            model_manager.initialize()

        if model_manager.summarizer:
            # Limit input to prevent issues with long texts
            input_text = text[:3000]
            summary = model_manager.summarizer(
                input_text,
                max_length=130,
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        else:
            # Fallback if summarizer failed to load
            sentences = text.split('.')
            return '. '.join(sentences[:3]) + '.'

    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        # Simple fallback - return first few sentences
        sentences = text.split('.')
        return '. '.join(sentences[:2]) + '.'


def extract_topics(text: str) -> List[str]:
    """Extract top keywords from text."""
    if not text or len(text.split()) < 20:
        return []

    try:
        model_manager = ModelManager()
        if not model_manager.initialized:
            model_manager.initialize()

        keywords = model_manager.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=3
        )
        return [kw[0] for kw in keywords]
    except Exception as e:
        logger.error(f"Topic extraction failed: {str(e)}")
        return []


def generate_tts(text: str) -> bytes:
    """Generate Hindi speech from text."""
    if not text:
        # Return empty audio bytes if no text
        empty_audio = np.zeros((1, 16000), dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, empty_audio, 16000, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()

    try:
        model_manager = ModelManager()
        if not model_manager.initialized:
            model_manager.initialize()

        if model_manager.tts_model and model_manager.tts_tokenizer:
            # Process input text
            inputs = model_manager.tts_tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                output = model_manager.tts_model(**inputs).waveform

            buffer = io.BytesIO()
            sf.write(buffer, output.numpy().T, model_manager.tts_model.config.sampling_rate, format='WAV')
            buffer.seek(0)
            return buffer.getvalue()
        else:
            # Fallback for missing TTS model
            empty_audio = np.zeros((1, 16000), dtype=np.float32)
            buffer = io.BytesIO()
            sf.write(buffer, empty_audio, 16000, format='WAV')
            buffer.seek(0)
            return buffer.getvalue()

    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        # Return empty audio in case of failure
        empty_audio = np.zeros((1, 16000), dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, empty_audio, 16000, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()