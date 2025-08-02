import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


def get_bert_embedding(text):
    """
    Returns the BERT embedding for a given text by averaging the token embeddings.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def extract_and_save_features(df, text_column, dataset_name, sample_size=None):
    """
    Extracts TF-IDF and BERT features separately and saves them as .npy files.
    """
    # If sample_size is provided, use only that many rows
    if sample_size is not None:
        df = df.head(sample_size)

    # Ensure text data is string and fill missing values
    texts = df[text_column].fillna("").astype(str).tolist()

    # ---- TF-IDF Features ----
    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
    np.save(f"{dataset_name}_tfidf.npy", tfidf_features)

    # ---- BERT Embeddings ----
    bert_features = np.array([get_bert_embedding(text) for text in texts])
    np.save(f"{dataset_name}_bert.npy", bert_features)

    print(f"✅ Features extracted and saved separately for {dataset_name}")
    print(f"TF-IDF Shape: {tfidf_features.shape} | BERT Shape: {bert_features.shape}")


# Define file paths for each dataset (Update paths as needed)
files = {
    "media_representation": "C:/Users/R.Parsad/Downloads/Datamedia/data/Lemmetization_labeled_media_representation.csv",
    "diversity_movies": "C:/Users/R.Parsad/Downloads/Datamedia/data/Lemmetization_labeled_diversity_movies.csv",
    "social_media_representation": "C:/Users/R.Parsad/Downloads/Datamedia/data/Lemmetization_labeled_social_media_representation.csv"
}

# Define the text column for each dataset (Update column names if needed)
text_columns = {
    "media_representation": "news_article_text",
    "diversity_movies": "community",
    "social_media_representation": "Social Media Platform"
}

# Process each dataset separately
for dataset_name, file_path in files.items():
    try:
        print(f"\nProcessing dataset: {dataset_name}")
        df = pd.read_csv(file_path)

        # Check if the specified text column exists
        col = text_columns.get(dataset_name, None)
        if col is None or col not in df.columns:
            print(f"❌ Text column for dataset '{dataset_name}' not found. Available columns: {df.columns.tolist()}")
            continue

        # Extract and save features separately for each dataset
        extract_and_save_features(df, text_column=col, dataset_name=dataset_name, sample_size=200)

    except Exception as e:
        print(f"❌ Error processing dataset '{dataset_name}': {e}")
