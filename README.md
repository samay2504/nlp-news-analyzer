# Company News Analyzer

A comprehensive tool that analyzes news articles for companies using Natural Language Processing techniques to extract sentiment, summarize content, identify key topics, and provide audio summaries in Hindi.

![News Analyzer](https://github.com/user-attachments/assets/d9ecbd5f-1074-4e9b-a66f-b4d8d962d467)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Components](#components)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Company News Analyzer is a full-stack application designed to help users gain insights from news articles about specific companies. The system fetches recent news articles, performs sentiment analysis, generates concise summaries, identifies key topics, and even provides audio summaries in Hindi. The analysis is presented through an intuitive Streamlit dashboard with interactive visualizations.

## Features

- **News Article Scraping**: Fetches the latest news articles for a specified company from Google News.
- **Sentiment Analysis**: Categorizes articles as Positive, Negative, or Neutral using VADER sentiment analysis.
- **Article Summarization**: Generates concise summaries using the BART large CNN model.
- **Topic Extraction**: Identifies key topics and themes across articles using KeyBERT.
- **Hindi Audio Summaries**: Generates spoken summaries in Hindi using a neural text-to-speech model.
- **Interactive Dashboard**: Presents the analysis through intuitive visualizations and filters.
- **REST API**: Backend API built with FastAPI for handling requests and processing data.

## Architecture

The application follows a client-server architecture:

1. **Frontend**: Streamlit web application that provides the user interface
2. **Backend**: FastAPI server that processes requests and handles NLP tasks
3. **NLP Pipeline**: Collection of models for sentiment analysis, summarization, topic extraction, and TTS

```
┌─────────────┐     REST API      ┌─────────────┐     ┌──────────────┐
│  Streamlit  │ ─────requests───> │   FastAPI   │ ─── │  News Source │
│  Frontend   │ <───responses──── │   Backend   │     └──────────────┘
└─────────────┘                   └─────────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │    NLP Pipeline     │
                              │  ┌───────────────┐  │
                              │  │   Sentiment   │  │
                              │  │   Analysis    │  │
                              │  └───────────────┘  │
                              │  ┌───────────────┐  │
                              │  │ Summarization │  │
                              │  └───────────────┘  │
                              │  ┌───────────────┐  │
                              │  │     Topic     │  │
                              │  │   Extraction  │  │
                              │  └───────────────┘  │
                              │  ┌───────────────┐  │
                              │  │   Hindi TTS   │  │
                              │  └───────────────┘  │
                              └─────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/samay2504/company-news-analyzer.git
   cd company-news-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data (for VADER sentiment analysis):
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## Usage

### Starting the Application

1. Start the API server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to http://localhost:8501 to access the application.

### Analyzing a Company

1. Enter a company name in the text input field.
2. Click the "Analyze" button.
3. The application will fetch and analyze news articles, then display the results in the dashboard.

## API Documentation

The FastAPI backend provides automatic API documentation. After starting the server, you can access:

- Interactive API documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

### Endpoints

- `GET /health`: Health check endpoint
- `POST /analyze`: Main endpoint for analyzing news about a company

Example request to the analyze endpoint:
```bash
curl -X 'POST' \
  'http://localhost:8000/analyze' \
  -H 'Content-Type: application/json' \
  -d '{"company_name": "Tesla"}'
```

## Components

### API (api.py)

The FastAPI backend that handles:
- Model initialization on startup
- Health check endpoint
- News analysis endpoint with company request processing
- Global exception handling

The API processes requests by:
1. Fetching news articles for the requested company
2. Analyzing sentiment of each article
3. Generating summaries of article content
4. Extracting key topics from each article
5. Creating a Hindi TTS audio summary
6. Returning the complete analysis to the frontend

### Frontend (app.py)

The Streamlit application that provides:
- User interface for entering company names
- API connection status monitoring
- Interactive visualizations of analysis results:
  - Sentiment distribution pie chart
  - Topic frequency bar chart
  - Audio playback for Hindi summary
- Article display with expandable details
- Tabs to filter articles by sentiment

### Utilities (utils.py)

Collection of utility functions and model management:
- `ModelManager`: Singleton class that initializes and manages NLP models
- `get_news_articles`: Fetches news articles from Google News
- `analyze_sentiment`: Determines sentiment using VADER
- `generate_summary`: Creates concise summaries using BART
- `extract_topics`: Identifies key topics using KeyBERT
- `generate_tts`: Produces Hindi audio using neural TTS

## Technologies Used

- **FastAPI**: High-performance web framework for building APIs
- **Streamlit**: Framework for building data apps
- **NLTK**: Natural language processing library (VADER sentiment analysis)
- **Hugging Face Transformers**: State-of-the-art NLP models
  - BART for summarization
  - VITS for Hindi text-to-speech
- **KeyBERT**: Keyword extraction using BERT embeddings
- **Plotly**: Interactive visualization library
- **Beautiful Soup**: Web scraping library
- **Feedparser**: RSS feed parsing
- **PyTorch**: Deep learning framework

## Project Structure

```
company-news-analyzer/
├── api.py              # FastAPI backend
├── app.py              # Streamlit frontend
├── utils.py            # Utility functions and model management
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── venv/               # Virtual environment (created during setup)
```

## Troubleshooting

### Common Issues

1. **API Not Connected Error**:
   - Ensure the FastAPI server is running on port 8000
   - Check for any error messages in the API terminal
   - Verify that no firewall is blocking the connection

2. **Model Loading Errors**:
   - Ensure you have enough disk space for model downloads
   - Check your internet connection for initial model downloads
   - Make sure you have sufficient RAM (at least 8GB recommended)

3. **No Articles Found**:
   - Try a different company name or check spelling
   - Ensure your internet connection allows access to Google News
   - Some companies might have limited recent news coverage

4. **Slow Performance**:
   - The first analysis might be slow due to model loading
   - Consider using a machine with GPU for faster processing
   - Reduce the number of articles processed by modifying the code

### Logs

Check the terminal running the FastAPI server for detailed logs about:
- Model initialization
- News article fetching
- Processing errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
