import streamlit as st
import requests
import base64
import json
import time
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="News Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stExpander {
        border-radius: 4px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    div.stButton > button:first-child {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Function to play audio
def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
    <audio controls>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)


# Function to check API health
def check_api_connection():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                return True
        return False
    except requests.exceptions.RequestException:
        return False


# Function to call the API with better error handling
def call_api(company):
    try:
        st.info(f"Sending request to analyze {company}...")
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"company_name": company},
            timeout=120  # 2 minute timeout
        )

        if response.status_code == 200:
            return response.json()
        else:
            # Try to parse error message
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", f"Error {response.status_code}")
                st.error(f"API Error: {error_msg}")
            except:
                st.error(f"API Error: Status code {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Make sure the API server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API server may be overloaded.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# Function to display articles
def display_articles(articles):
    for i, art in enumerate(articles, 1):
        with st.expander(f"Article {i}: {art['title']}"):
            st.write(f"**Summary:** {art['summary']}")

            # Create columns for metadata
            col1, col2 = st.columns(2)

            # Sentiment with color coding
            sentiment_color = {
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'gray'
            }.get(art['sentiment'], 'gray')

            col1.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{art['sentiment']}</span>",
                          unsafe_allow_html=True)

            # Topics
            col2.write(f"**Topics:** {', '.join(art['topics']) if art['topics'] else 'None'}")

            # Link to original article
            if 'link' in art:
                st.markdown(f"**Source:** [Read Original Article]({art['link']})")


# Main application function
def main():
    # Sidebar for settings
    with st.sidebar:
        st.title("📊 Settings")

        # API connection status
        api_connected = check_api_connection()
        if api_connected:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Connected")
            st.error("""
            Please start the API server with:
            ```
            uvicorn api:app --host 0.0.0.0 --port 8000 --reload
            ```
            """)

        # Language selection (for future implementation)
        language = st.selectbox("Summary Language", ["Hindi", "English"], index=0)

        # Analysis depth (for future implementation)
        analysis_depth = st.slider("Analysis Depth", 1, 10, 5)

        st.markdown("---")
        st.info("This app analyzes news sentiment for companies using NLP techniques.")

    # Main content
    st.title("📰 Company News Analyzer")
    st.write("Enter a company name to analyze recent news articles and sentiment")

    # Input form
    company = st.text_input("Company Name", "Tesla")

    if st.button("Analyze"):
        if not api_connected:
            st.error("API server is not connected. Please start the API server first.")
            return

        if not company:
            st.warning("Please enter a company name")
            return

        # Fetch and display analysis
        with st.spinner("Analyzing news articles..."):
            data = call_api(company)

            if not data:
                st.error("Failed to get analysis results.")
                return

            # Display analysis results
            st.header(f"Analysis Report for {data['company']}")

            # Create layout
            col1, col2 = st.columns([2, 1])

            # Left column - Sentiment Analysis
            with col1:
                st.subheader("📊 Sentiment Analysis")

                # Pie chart for sentiment
                fig = px.pie(
                    values=list(data['sentiment_distribution'].values()),
                    names=list(data['sentiment_distribution'].keys()),
                    title="Sentiment Distribution",
                    color_discrete_sequence=['#00cc96', '#ef553b', '#636efa'],
                    hole=0.4
                )
                fig.update_traces(textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

                # Metrics row
                metrics = data['sentiment_distribution']
                metric_cols = st.columns(3)
                metric_cols[0].metric("Positive", metrics['Positive'])
                metric_cols[1].metric("Neutral", metrics['Neutral'])
                metric_cols[2].metric("Negative", metrics['Negative'])

            # Right column - Topics and Audio
            with col2:
                st.subheader("🔑 Key Topics")

                # Display common topics
                if data['common_topics']:
                    topics_data = []
                    for topic in data['common_topics']:
                        count = sum(1 for art in data['articles'] if topic in art['topics'])
                        topics_data.append({"topic": topic, "count": count})

                    if topics_data:
                        fig = px.bar(
                            x=[t["topic"] for t in topics_data],
                            y=[t["count"] for t in topics_data],
                            labels={"x": "Topic", "y": "Count"},
                            title="Common Topics"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No common topics found")

                # Audio summary
                if 'audio' in data and data['audio']:
                    st.subheader("🎧 Hindi Audio Summary")
                    try:
                        audio_bytes = bytes.fromhex(data['audio'])
                        st.audio(audio_bytes, format='audio/wav')
                    except Exception as e:
                        st.error(f"Failed to play audio: {e}")

            # Articles section with tabs
            if 'articles' in data and data['articles']:
                st.subheader("📰 News Articles Analysis")

                articles = data['articles']
                pos_articles = [a for a in articles if a['sentiment'] == 'Positive']
                neg_articles = [a for a in articles if a['sentiment'] == 'Negative']
                neu_articles = [a for a in articles if a['sentiment'] == 'Neutral']

                tabs = st.tabs(["All Articles", "Positive", "Negative", "Neutral"])

                with tabs[0]:
                    display_articles(articles)

                with tabs[1]:
                    if pos_articles:
                        display_articles(pos_articles)
                    else:
                        st.info("No positive articles found")

                with tabs[2]:
                    if neg_articles:
                        display_articles(neg_articles)
                    else:
                        st.info("No negative articles found")

                with tabs[3]:
                    if neu_articles:
                        display_articles(neu_articles)
                    else:
                        st.info("No neutral articles found")
            else:
                st.warning("No articles found for analysis")


if __name__ == "__main__":
    main()

# Run with: streamlit run app.py