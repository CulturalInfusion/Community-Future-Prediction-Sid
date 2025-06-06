import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/R.Parsad/Downloads/Datamedia/data/Lemmetization_labeled_social_media_representation.csv"  # Update this path if needed
social_media_df = pd.read_csv(file_path)

# Convert percentage columns to numeric
for col in ["Positive Representation (%)", "Neutral Representation (%)", "Negative Representation (%)"]:
    social_media_df[col] = pd.to_numeric(social_media_df[col], errors='coerce')

# Set visualization style
sns.set_theme(style="whitegrid")

# 1️⃣ Representation Trends Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=social_media_df, x="Year", y="Positive Representation (%)", hue="Community", marker="o")
plt.title("📈 Positive Representation Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Positive Representation (%)")
plt.legend(title="Community", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.savefig("C:/Users/R.Parsad/Downloads/Datamedia/data")  # Save as PNG
plt.show()

# 2️⃣ Distribution of Representation Across Platforms
plt.figure(figsize=(12, 6))
sns.boxplot(data=social_media_df, x="Social Media Platform", y="Positive Representation (%)", palette="Set2")
plt.title("📊 Positive Representation Distribution by Platform")
plt.xlabel("Social Media Platform")
plt.ylabel("Positive Representation (%)")
plt.grid()
plt.savefig("C:/Users/R.Parsad/Downloads/Datamedia/data")  # Save as PNG
plt.show()

# 3️⃣ Sentiment Trends (Positive vs. Negative) Across Communities
plt.figure(figsize=(12, 6))
sns.barplot(data=social_media_df, x="Community", y="Positive Representation (%)", color="green", label="Positive")
sns.barplot(data=social_media_df, x="Community", y="Negative Representation (%)", color="red", alpha=0.7, label="Negative")
plt.xticks(rotation=90)
plt.title("✅ Positive vs. ❌ Negative Representation by Community")
plt.xlabel("Community")
plt.ylabel("Representation (%)")
plt.legend()
plt.grid()
plt.savefig("C:/Users/R.Parsad/Downloads/Datamedia/data")  # Save as PNG
plt.show()

# 4️⃣ Total Mentions Across Social Media Platforms
plt.figure(figsize=(10, 6))
sns.barplot(data=social_media_df, x="Social Media Platform", y="Total Mentions", estimator=sum, palette="coolwarm")
plt.title("📌 Total Mentions Across Social Media Platforms")
plt.xlabel("Platform")
plt.ylabel("Total Mentions")
plt.grid()
plt.savefig("C:/Users/R.Parsad/Downloads/Datamedia/data")  # Save as PNG
plt.show()

# Save cleaned dataset
cleaned_path = "C:/Users/R.Parsad/Downloads/Datamedia/data"
social_media_df.to_csv(cleaned_path, index=False)
print(f"✅ Cleaned dataset saved to: {cleaned_path}")
