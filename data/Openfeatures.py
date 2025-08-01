import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_npy_file(file_path):
    """
    Loads an .npy file, prints metadata, and visualizes the content if it's numerical.
    """
    try:
        # Ensure the file exists before loading
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)

        # Print metadata
        print(f"\n✅ Successfully loaded: {file_path}")
        print("Shape:", data.shape)
        print("Data Type:", data.dtype)

        # Preview data (first 5 rows if applicable)
        if data.ndim > 1:  # If it's a matrix
            print("First 5 rows:\n", data[:5])
        else:  # If it's a 1D array
            print("First 10 elements:\n", data[:10])

        # If the data is numerical, visualize it
        if data.ndim == 2 and data.shape[1] < 50:  # Limit to 50 features for readability
            plt.figure(figsize=(12, 6))
            sns.heatmap(data[:50], cmap="coolwarm", annot=False)  # Plot first 50 rows
            plt.title(f"Feature Matrix Heatmap: {os.path.basename(file_path)}")
            plt.xlabel("Feature Index")
            plt.ylabel("Sample Index")
            plt.show()
        elif data.ndim == 1:
            plt.figure(figsize=(10, 4))
            plt.plot(data[:100])  # Plot first 100 elements if it's a sequence
            plt.title(f"1D Data Plot: {os.path.basename(file_path)}")
            plt.show()

        return data

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# ✅ List of .npy files to load (Replace with actual file paths)
npy_files = [
    "media_representation_tfidf.npy",
    "media_representation_bert.npy",
    "diversity_movies_tfidf.npy",
    "diversity_movies_bert.npy",
    "social_media_representation_tfidf.npy",
    "social_media_representation_bert.npy"
]

# ✅ Load and inspect multiple .npy files
loaded_data = {}  # Dictionary to store loaded data

for file_path in npy_files:
    loaded_data[file_path] = load_npy_file(file_path)

print("\n✅ All .npy files processed successfully!")
