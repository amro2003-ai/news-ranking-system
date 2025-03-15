from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import pandas as pd
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the main directory path
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing required modules from custom packages
from model.sentimentModel import Model
from rankingSystem.rankingSystemPipeline import RankingSystemPipeline
from config.configLoder import ConfigLoder

class SentimentAPI:
    def __init__(self):
        """
        Initializes the FastAPI application, loads the sentiment model,
        and configures Cross-Origin Resource Sharing (CORS).
        """
        self.app = FastAPI()
        self.model = Model()
        self.config = ConfigLoder().load_config()

        logger.info("Initializing Sentiment API...")
        
        # Enable CORS for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], 
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        logger.info("CORS middleware configured.")

        # Define API routes
        self.define_routes()

    def define_routes(self):
        """
        Defines the API routes.
        """
        @self.app.get("/")
        async def get():
            """Returns the HTML response from a template file."""
            try:
                template = self.config['template']['template_html']
                with open(template) as f:
                    html_content = f.read()
                logger.info("Served HTML template successfully.")
                return HTMLResponse(content=html_content, status_code=200)
            except Exception as e:
                logger.error(f"Error loading template: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Template error: {str(e)}")

        @self.app.post("/predict/")
        async def predict_sentiment(request: Model.SentimentRequest):
            """Predicts sentiment based on input request."""
            try:
                logger.info("Received sentiment prediction request.")
                return self.model._predict_sentiment(request)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        @self.app.get("/ranking/")
        async def get_rank():
            """Fetches and returns the top-ranked news data."""
            try:
                logger.info("Processing ranking request.")
                news_csv_path = self.config['data']['news_csv']
                ranking_pipeline = RankingSystemPipeline(news_csv_path)
                ranked_data = ranking_pipeline.sortData()

                # Ensure ranked_data is a DataFrame
                if not isinstance(ranked_data, pd.DataFrame):
                    logger.error("Ranking system did not return a valid DataFrame.")
                    raise ValueError("getRankScore() did not return a valid DataFrame")
                
                # Select top-ranked news & convert to JSON
                top_n = self.config['ranking']['top_n']
                result = ranked_data.head(top_n).to_dict(orient="records")
                logger.info("Ranking request processed successfully.")
                return result

            except Exception as e:
                logger.error(f"Ranking error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Ranking error: {str(e)}")

    def run(self):
        """
        Starts the FastAPI server using Uvicorn.
        """
        api_host = self.config['api']['host']
        api_port = int(self.config['api']['port'])
        logger.info(f"Starting API server at {api_host}:{api_port}...")
        uvicorn.run(self.app, host=api_host, port=api_port)
