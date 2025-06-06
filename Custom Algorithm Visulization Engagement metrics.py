import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

# ✅ Define extracted feature files (TF-IDF & BERT)
files = {
    "media_representation": {
        "tfidf": "media_representation_tfidf.npy",
        "bert": "media_representation_bert.npy"
    },
    "diversity_movies": {
        "tfidf": "diversity_movies_tfidf.npy",
        "bert": "diversity_movies_bert.npy"
    },
    "social_media_representation": {
        "tfidf": "social_media_representation_tfidf.npy",
        "bert": "social_media_representation_bert.npy"
    }
}

metrics_results = {}

# ✅ Process each dataset
for dataset, file_paths in files.items():
    print(f"\nProcessing: {dataset}")

    # Check if files exist before loading
    if not os.path.exists(file_paths["tfidf"]) or not os.path.exists(file_paths["bert"]):
        print(f"❌ Missing files for {dataset}. Skipping...")
        continue

    # ✅ Load TF-IDF and BERT feature matrices separately
    tfidf_features = np.load(file_paths["tfidf"], allow_pickle=True)
    bert_features = np.load(file_paths["bert"], allow_pickle=True)

    # ---- 1️⃣ Text Complexity Score (TF-IDF Variance) ----
    complexity_score = np.var(tfidf_features, axis=1).mean()

    # ---- 2️⃣ Semantic Diversity Score (Cosine Similarity of BERT Embeddings) ----
    similarity_matrix = cosine_similarity(bert_features)
    semantic_diversity_score = 1 - similarity_matrix.mean()  # Higher = more diverse

    # ---- 3️⃣ Sentiment Polarity Distribution (Mean of BERT Features) ----
    sentiment_polarity = np.mean(bert_features, axis=1).mean()  # Higher = More positive

    # ---- 4️⃣ Topic Density (PCA on TF-IDF) ----
    pca = PCA(n_components=5)
    pca.fit(tfidf_features)
    topic_density = sum(pca.explained_variance_ratio_)  # % variance explained by top 10 topics

    # ---- 5️⃣ Community Representation Index (BERT Feature Spread) ----
    community_rep_index = np.mean(np.std(bert_features, axis=0))  # Higher = More balanced representation

    # ✅ Store results
    metrics_results[dataset] = {
        "Text Complexity Score": complexity_score,
        "Semantic Diversity Score": semantic_diversity_score,
        "Sentiment Polarity Score": sentiment_polarity,
        "Topic Density Score": topic_density,
        "Community Representation Index": community_rep_index
    }

# ✅ Convert results to DataFrame
metrics_df = pd.DataFrame.from_dict(metrics_results, orient="index")

# ✅ Save as a CSV report
metrics_df.to_csv("engagement_metrics_report.csv", index=True)

# ✅ Visualize Metrics
plt.figure(figsize=(12, 6))
metrics_df.plot(kind="bar", figsize=(12, 6), colormap="coolwarm", width=0.7)
plt.title("Engagement-Based Metrics for Media Representation")
plt.xlabel("Dataset")
plt.ylabel("Metric Score")
plt.xticks(rotation=45)
plt.grid()
plt.legend(loc="best")
plt.savefig("engagement_metrics_visual.png", bbox_inches="tight", dpi=300)
plt.show()

print("\n✅ Engagement metrics calculated and saved successfully!")
