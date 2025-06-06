import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load the dataset
news_df = pd.read_csv("C:/Users/R.Parsad/Downloads/Datamedia/data/Lemmetization_labeled_media_representation.csv")

# Handle missing values
news_df.fillna(news_df.median(numeric_only=True), inplace=True)
news_df.fillna(news_df.mode().iloc[0], inplace=True)

# Function to calculate sentiment score
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Apply sentiment analysis
news_df["sentiment_score"] = news_df["news_article_text"].apply(get_sentiment)

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.histplot(news_df["sentiment_score"], bins=20, kde=True, color="purple")
plt.title("Sentiment Score Distribution in News Articles")
plt.xlabel("Sentiment Score (-1 Negative to +1 Positive)")
plt.ylabel("Count")
plt.grid()
plt.show()

# Save cleaned dataset
news_df.to_csv("C:/Users/R.Parsad/Downloads/Datamedia/data", index=False)
