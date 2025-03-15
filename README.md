# News Ranking System

This project analyzes financial news with the aim of improving the decision-making process for investors. An LSTM model was built to classify news as positive or negative, and a ranking system was developed to process the latest news and rank the best stocks to invest in. A user interface was also created to interact with the model and display the top five most influential news items.

**Features:**

- **NLP:** An LSTM model was built to classify news as positive or negative.
- **Ranking News Pipeline:** A pipeline was developed to read data, analyze it, and provide the most impactful news.
- **FastAPI:** FastAPI was used for backend development.
- **User Interface:** A user interface was created to interact with the model and display the top 5 most influential news items.

**Prerequisites:**  
1. Python (>=3.8)  
2. Visual Studio Code

**Instructions:**  
1. Install dependencies using:  
   ```bash
   pip install -r requirements.txt
   ```
   
2. Open the UI in a browser using:  
   [http://127.0.0.1:8000](http://127.0.0.1:8000)


## Project Structure

assets/
    ui.png
config/
    config.yaml
data/
    news.csv
model/
    label_encoder.pkl
    sentiment_model.h5
    tokenizer.pkl
notebook/
    news-sentiment-analysis.ipynb
src/
    api/
        template/
            interface.html
        api.py
        __init__.py
    model/
        sentimentModel.py
        __init__.py
    config/
        configLoader.py
        __init__.py
    rankingSystem/
        rankingSystemPipeline.py
        __init__.py
main.py
requirements.txt


