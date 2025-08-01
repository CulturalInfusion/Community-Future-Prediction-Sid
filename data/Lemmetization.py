import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLTK lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and lemmatize text using NLTK
def preprocess_text(text):
    if isinstance(text, str):  # Ensure input is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize & remove stopwords
        return " ".join(tokens)
    return text

# Load datasets
movies_df = pd.read_excel("C:/Users/R.Parsad/Downloads/labeled_diversity_movies.xlsx")
social_media_df = pd.read_excel("C:/Users/R.Parsad/Downloads/labeled_social_media_representation.xlsx")
news_df = pd.read_excel("C:/Users/R.Parsad/Downloads/labeled_media_representation.xlsx")

# Apply preprocessing column-wise
for col in movies_df.select_dtypes(include=['object']).columns:
    movies_df[col] = movies_df[col].map(preprocess_text)

for col in social_media_df.select_dtypes(include=['object']).columns:
    social_media_df[col] = social_media_df[col].map(preprocess_text)

for col in news_df.select_dtypes(include=['object']).columns:
    news_df[col] = news_df[col].map(preprocess_text)

# Save processed datasets
movies_df.to_excel("Lemmetization_labeled_diversity_movies.xlsx", index=False)
social_media_df.to_excel("Lemmetization_labeled_social_media_representation.xlsx", index=False)
news_df.to_excel("Lemmetization_labeled_media_representation.xlsx", index=False)

print("Preprocessing completed. Cleaned files have been saved.")
