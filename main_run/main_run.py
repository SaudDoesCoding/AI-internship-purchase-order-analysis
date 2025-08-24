#importing neccessary libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import nltk 
from nltk.corpus import stopwords
import spacy
from langdetect import detect
import arabic_reshaper
from bidi.algorithm import get_display
import regex as re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Loading Data

data = pd.read_excel("purchase-order-items.xlsx")

print("Shape:", data.shape)
data.head()

#Check if there's NaNs 

print(data.isna().sum)

data = data.dropna(subset=["Item Name"])

# Ensure that the bcy is all numeric 
data["Total Bcy"] = pd.to_numeric(data["Total Bcy"], errors='coerce')
data = data.dropna(subset=["Total Bcy"])

#Language detector
def detect_lang_group(text):
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    has_ar = bool(re.search(r'[\u0600-\u06FF]', text))
    has_en = bool(re.search(r'[A-Za-z]', text))
    if has_ar:
        return "ar"
    elif has_en:
        return "en"
    else:
        return "other"

data["lang_group"] = data["Item Name"].apply(detect_lang_group)
print(data["lang_group"].value_counts())


def cluster_group(df, col="Item Name", n_clusters=5):
    vectorizer = TfidfVectorizer(
        analyzer="char",   # works well for Arabic & English
        ngram_range=(3,4),
        max_features=500
    )
    X = vectorizer.fit_transform(df[col])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    df["cluster"] = kmeans.fit_predict(X)
    return df

# Arabic clusters
arabic_df = cluster_group(data[data["lang_group"] == "ar"].copy(), n_clusters=6)

# English clusters
english_df = cluster_group(data[data["lang_group"] == "en"].copy(), n_clusters=6)

# Combine results
clustered = pd.concat([arabic_df, english_df])

# Next section is only used to ensure that the clusters are working and following rules. 
# for i in range(6):
  #  print(f"\nArabic Cluster {i}:")
   # print(arabic_df[arabic_df["cluster"] == i]["Item Name"].head(10).to_string())

#for i in range(6):
 #   print(f"\nEnglish Cluster {i}:")
  #  print(english_df[english_df["cluster"] == i]["Item Name"].head(10).to_string())

# Grouping clusters with sum spend 
arabic_spend = arabic_df.groupby("cluster")["Total Bcy"].sum()
english_spend = english_df.groupby("cluster")["Total Bcy"].sum()

print("\nArabic Spend per Cluster:")
print(arabic_spend)

print("\nEnglish Spend per Cluster:")
print(english_spend)

# Visualizing the clusters x Sum spent  

fig, axes = plt.subplots(1, 2, figsize=(16,5))  # 1 row, 2 columns

# Arabic clusters
arabic_spend.plot(kind="bar", ax=axes[0], color="skyblue", title="Arabic Spend by Cluster")
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("Total Bcy")
axes[0].grid(axis='y')

# English clusters
english_spend.plot(kind="bar", ax=axes[1], color="lightgreen", title="English Spend by Cluster")
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Total Bcy")
axes[1].grid(axis='y')

plt.tight_layout()
plt.show()

#Analysis of the figures 

print("Arabic Cluster Counts:")
print(arabic_df["cluster"].value_counts())
print("\nEnglish Cluster Counts:")
print(english_df["cluster"].value_counts()) 

# most Arabic spend cluster
top_arabic_cluster = arabic_spend.idxmax()
print(f"Arabic cluster with highest spend: {top_arabic_cluster}, Amount: {arabic_spend.max()}")

# most English spend cluster
top_english_cluster = english_spend.idxmax()
print(f"English cluster with highest spend: {top_english_cluster}, Amount: {english_spend.max()}")

# Seeing what are the most spent items 

print("Top Arabic cluster items:")
print(arabic_df[arabic_df["cluster"] == top_arabic_cluster][["Item Name", "Total Bcy"]].head(10))

print("\nTop English cluster items:")
print(english_df[english_df["cluster"] == top_english_cluster][["Item Name", "Total Bcy"]].head(10))


# Counting the top 10 cluster sizes 
arabic_counts = arabic_df["cluster"].value_counts()
english_counts = english_df["cluster"].value_counts()

print("Arabic Cluster Sizes (top 10):")
print(arabic_counts.head(10))

print("\nEnglish Cluster Sizes (top 10):")
print(english_counts.head(10))

# Calculating the average spend per item per cluster 

arabic_avg = arabic_df.groupby("cluster")["Total Bcy"].mean()
english_avg = english_df.groupby("cluster")["Total Bcy"].mean()

print("Arabic Cluster Average Spend per Item:")
print(arabic_avg)

print("\nEnglish Cluster Average Spend per Item:")
print(english_avg)

# Making a summary of the results for better inspection 

arabic_summary = arabic_df.groupby("cluster").agg(
    cluster_size=("Item Name", "count"),
    total_spend=("Total Bcy", "sum"),
    avg_spend=("Total Bcy", "mean")
).sort_values(by="total_spend", ascending=False)

english_summary = english_df.groupby("cluster").agg(
    cluster_size=("Item Name", "count"),
    total_spend=("Total Bcy", "sum"),
    avg_spend=("Total Bcy", "mean")
).sort_values(by="total_spend", ascending=False)

print("Arabic Cluster Summary:")
print(arabic_summary)
print("\nEnglish Cluster Summary:")
print(english_summary)

fig, axes = plt.subplots(2, 3, figsize=(18,10))

# Arabic plots
arabic_summary["total_spend"].plot(kind="bar", ax=axes[0,0], color="skyblue", title="Arabic: Total Spend")
axes[0,0].set_ylabel("Total Bcy")

arabic_summary["avg_spend"].plot(kind="bar", ax=axes[0,1], color="orange", title="Arabic: Avg Spend per Item")
axes[0,1].set_ylabel("Bcy per Item")

arabic_summary["cluster_size"].plot(kind="bar", ax=axes[0,2], color="green", title="Arabic: Cluster Size")
axes[0,2].set_ylabel("Number of Items")

# English plots
english_summary["total_spend"].plot(kind="bar", ax=axes[1,0], color="lightgreen", title="English: Total Spend")
axes[1,0].set_ylabel("Total Bcy")

english_summary["avg_spend"].plot(kind="bar", ax=axes[1,1], color="salmon", title="English: Avg Spend per Item")
axes[1,1].set_ylabel("Bcy per Item")

english_summary["cluster_size"].plot(kind="bar", ax=axes[1,2], color="purple", title="English: Cluster Size")
axes[1,2].set_ylabel("Number of Items")

plt.tight_layout()
plt.show()

top_arabic_cluster = arabic_summary.index[0]
print("\nTop Arabic cluster items:")
print(arabic_df[arabic_df["cluster"] == top_arabic_cluster].sort_values(by="Total Bcy", ascending=False).head(10))

top_english_cluster = english_summary.index[0]
print("\nTop English cluster items:")
print(english_df[english_df["cluster"] == top_english_cluster].sort_values(by="Total Bcy", ascending=False).head(10))