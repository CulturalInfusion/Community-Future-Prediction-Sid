import pandas as pd
import numpy as np
import re
import spacy
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler

# Download required NLTK resources
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# ============================
# 📌 STEP 1: TEXT PREPROCESSING
# ============================
def preprocess_text(text):
    """Clean, tokenize, and lemmatize text data."""
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = word_tokenize(text)  # Tokenization
    tokens = [nlp(word)[0].lemma_ for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# ============================
# 📌 STEP 2: LOAD DATASET
# ============================
df = pd.read_csv("")  # Replace with actual dataset
df["processed_text"] = df["text"].apply(preprocess_text)

# ============================
# 📌 STEP 3: TF-IDF FEATURE EXTRACTION
# ============================
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()  # Important words

# ============================
# 📌 STEP 4: WORD2VEC MAPPING (TF-IDF → Word2Vec)
# ============================
tokenized_sentences = [text.split() for text in df["processed_text"]]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=5, min_count=1, workers=4)

def get_tfidf_word2vec_embedding(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)

word2vec_features = np.array([get_tfidf_word2vec_embedding(text) for text in df["processed_text"]])

# ============================
# 📌 STEP 5: BERT FEATURE EXTRACTION (Deep Context-Aware Embeddings)
# ============================
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    """Generate BERT embeddings for input text"""
    tokens = bert_tokenizer(text, padding="max_length", max_length=50, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

bert_features = np.array([get_bert_embedding(text) for text in df["processed_text"]])

# ============================
# 📌 STEP 6: CONCATENATE ALL FEATURES
# ============================
combined_features = np.concatenate([tfidf_matrix.toarray(), word2vec_features, bert_features], axis=1)

# ============================
# 📌 STEP 7: STANDARDIZATION (Optional)
# ============================
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# ============================
# 📌 FINAL: READY FOR TRAINING
# ============================
np.save("media_representation_features.npy", scaled_features)
