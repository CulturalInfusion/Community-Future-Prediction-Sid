import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
tv_df = pd.read_csv("C:/Users/R.Parsad/Downloads/Datamedia/data/Lemmetization_labeled_diversity_movies.csv")

# Handle missing values
tv_df.fillna(tv_df.median(numeric_only=True), inplace=True)
tv_df.fillna(tv_df.mode().iloc[0], inplace=True)

# Plot representation trends over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=tv_df, x="year", y="representation_percentage", hue="community", marker="o")
plt.title("Diversity Representation in TV Shows Over Time")
plt.xlabel("Year")
plt.ylabel("Representation (%)")
plt.legend(title="Community")
plt.grid()
plt.show()

# Save cleaned dataset
tv_df.to_csv("C:/Users/R.Parsad/Downloads/Datamedia/data", index=False)
