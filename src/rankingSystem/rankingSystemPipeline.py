import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RankingSystemPipeline:
    def __init__(self, path):
        """
        Initialize the Ranking System Pipeline.
        
        param path: str - Path to the CSV file containing news sentiment data.
        task: Reads the CSV file into a DataFrame and computes rank scores.
        """
        self.path = path
        logging.info(f"Loading dataset from: {self.path}")
        self.df = pd.read_csv(self.path)
        self.getRankScore()

    def getRankScore(self):
        """
        Compute ranking scores based on sentiment analysis data.
        
        task: Calculates a rank score using a weighted sum of sentiment scores.
        raises ValueError: If required columns are missing from the DataFrame.
        """
        logging.info("Computing rank scores...")
        
        # Ensure all required columns exist
        required_columns = ["pos", "neu", "neg", "compound", "date", "news"]
        for col in required_columns:
            if col not in self.df.columns:
                logging.error(f"Missing required column: {col}")
                raise ValueError(f"Missing required column: {col}")
        
        # Compute rank score
        self.df["rank_score"] = (
            (self.df["pos"] * 2) +  # Positive sentiment weight
            (self.df["neu"] * 1) -  # Neutral sentiment weight
            (self.df["neg"] * 3) +  # Negative sentiment penalty
            (self.df["compound"] * 5)  # Compound sentiment weight
        )
        logging.info("Rank scores computed successfully.")

    def sortData(self):
        """
        Sort the dataset based on rank scores in descending order.
        
        task: Sorts news articles based on their computed rank score.
        return: pd.DataFrame - DataFrame containing sorted news articles with date, news, and rank_score columns.
        """
        logging.info("Sorting data by rank score...")
        
        if "rank_score" not in self.df.columns:
            logging.warning("Rank scores not found. Recomputing...")
            self.getRankScore()
        
        df_sorted = self.df.sort_values(by="rank_score", ascending=False)
        logging.info("Data sorted successfully.")
        
        return df_sorted[["date", "news", "rank_score"]]
