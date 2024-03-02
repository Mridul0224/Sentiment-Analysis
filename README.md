# Sentiment Analysis in Python

This Python script performs sentiment analysis on a dataset of reviews using various Natural Language Processing (NLP) techniques and models, including NLTK (Natural Language Toolkit) and transformers like VADER (Valence Aware Dictionary and sEntiment Reasoner) and RoBERTa (Robustly Optimized BERT Pretraining Approach).

## Features

- **Data Reading and Exploration**: Utilizes pandas for data manipulation and matplotlib, seaborn for data visualization.
- **Sentiment Analysis Techniques**:
  - **NLTK Basics**: Tokenization, POS tagging, Named Entity Recognition.
  - **VADER Sentiment Scoring**: For obtaining negative, neutral, and positive sentiment scores.
  - **RoBERTa Pretrained Model**: Utilizing a model trained on a large corpus for sentiment analysis.
  - **Comparison of Sentiment Scores**: Between VADER and RoBERTa models.
- **BERT Sentiment Analysis**: Additional sentiment analysis using a BERT model for multilingual sentiment analysis.
- **Visualization**: Comparative visualization of sentiment analysis results.

## Requirements

- Python 3.x
- NLTK
- pandas
- numpy
- matplotlib
- seaborn
- transformers
- tqdm
- scipy
- requests (for web scraping functionality)
- beautifulsoup4 (for web scraping functionality)

## Installation

Ensure you have Python installed, then run the following commands to install the necessary libraries:

```bash
pip install nltk pandas numpy matplotlib seaborn transformers tqdm scipy requests beautifulsoup4
```

## Usage
To use the Sentiment Analysis script, follow these steps:

Update the dataset path in the script to point to your dataset of reviews.
Execute the script with Python by running the following command in your terminal:
```bash
python Sentiment_Analysis.py
```
The script processes the dataset and outputs sentiment analysis results, including visualizations of the sentiments.

## Dataset
For the Sentiment Analysis script to function correctly, your dataset should adhere to the following requirements:

The dataset must be in a format that the script can read (e.g., CSV, Excel, or JSON).
webscrapping method can also be used, proper modules must be installed for the function.
It should contain at least one column for review texts and another for their corresponding scores or ratings, which the script will use to perform sentiment analysis.
Ensure your dataset is prepared according to these guidelines before running the script.



