from src.API.api import SentimentAPI

try:
    sentiment_api = SentimentAPI()
    sentiment_api.run()
except Exception as e:
    print(f"Error starting the API: {e}")