import pandas as pd
import re
from transformers import pipeline
import os
# Load Excel dataset
input_file ="C:/Users/R.Parsad/Downloads/Datamedia/media_representation_past_20_years.xlsx"+"C:/Users/R.Parsad/Downloads/Datamedia/cleaned_social_media_representation_past_20_years.csv"+"C:/Users/R.Parsad/Downloads/Datamedia/Cleaned_Diversity_movies_dataset (2).xlsx"
df = pd.read_excel(input_file)
print("Column Names in Excel File:", df.columns)
# Ensure column names
if 'article_content' not in df.columns or 'date_published' not in df.columns or 'media_source' not in df.columns:
    raise ValueError("Ensure your dataset has 'article_content', 'date_published', and 'media_source' columns.")

# Preprocess text
def clean_text(text):
    return re.sub(r'\W+', ' ', str(text)).lower()

df['clean_text'] = df['article_content'].apply(clean_text)

# Extract year from date
df['year'] = pd.to_datetime(df['date_published'], errors='coerce').dt.year

# Load Sentiment Analysis Model
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to classify sentiment
def classify_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])  # Limit to 512 tokens
        return result[0]['label']
    except:
        return "NEUTRAL"

df['sentiment'] = df['clean_text'].apply(classify_sentiment)

# Group by Year
summary = df.groupby(['year', 'media_source']).agg(
    total_articles=('article_content', 'count'),
    positive_sentiment=('sentiment', lambda x: (x == "POSITIVE").sum()),
    negative_sentiment=('sentiment', lambda x: (x == "NEGATIVE").sum()),
    neutral_sentiment=('sentiment', lambda x: (x == "NEUTRAL").sum())
).reset_index()

# Save labeled dataset to Excel
output_file = "labeled_media_data.xlsx"
summary.to_excel(output_file, index=False)

print(f"Labeled dataset saved as {output_file}")
print(f"labelling datasets for to be used in labelling the dataset")
