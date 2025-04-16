


import os
import pandas as pd
import numpy as np
#import streamlit as st
from dotenv import load_dotenv
import openai
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
from tqdm import tqdm
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()



def clean_text(text):
    cleaned_text = []
    for text in text:
        if not isinstance(text, str):  # Handle NaN or float values
            text = ""
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters, punctuation, numbers
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stopwords.words('english')]
        cleaned_text.append(" ".join(tokens))
    return cleaned_text


# Normalize
def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x if norm == 0 else x / norm
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

# Token chunker with skipped tracking
def chunk_by_tokens_with_counts(text_list, max_tokens=8000, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    batches, batch_token_counts, skipped_texts = [], [], []
    current_batch, current_counts = [], []
    current_tokens = 0

    for idx, text in enumerate(text_list):
        text = str(text).strip()
        if not text:
            continue
        tokens = encoding.encode(text)
        token_len = len(tokens)

        if token_len > max_tokens:
            skipped_texts.append({
                "index": idx, "text": text, "token_count": token_len, "reason": "Too many tokens"
            })
            continue

        if current_tokens + token_len > max_tokens:
            batches.append(current_batch)
            batch_token_counts.append(current_counts)
            current_batch, current_counts = [text], [token_len]
            current_tokens = token_len
        else:
            current_batch.append(text)
            current_counts.append(token_len)
            current_tokens += token_len

    if current_batch:
        batches.append(current_batch)
        batch_token_counts.append(current_counts)

    return batches, batch_token_counts, skipped_texts

# OpenAI embedding function
def get_openai_embeddings(text_list, model="text-embedding-3-small", cut_dim=256, normalize=True):
    all_embeddings, all_token_counts, all_skipped = [], [], []
    batches, batch_token_counts, batch_skipped = chunk_by_tokens_with_counts(text_list, model=model)
    all_skipped.extend(batch_skipped)

    for batch_idx, (batch, counts) in enumerate(zip(batches, batch_token_counts)):
        try:
            response = openai.embeddings.create(input=batch, model=model, encoding_format="float")
            if len(response.data) != len(batch):
                print(f"⚠️ Batch {batch_idx} mismatch. Skipping batch.")
                for i, text in enumerate(batch):
                    all_skipped.append({"index": None, "text": text, "token_count": counts[i], "reason": "API mismatch"})
                continue

            for i, item in enumerate(response.data):
                vec = item.embedding[:cut_dim]
                if normalize:
                    vec = normalize_l2(vec)
                all_embeddings.append(vec)
                all_token_counts.append(counts[i])

        except openai.error.OpenAIError as e:
            print(f"❌ API error on batch {batch_idx}: {e}")
            for i, text in enumerate(batch):
                all_skipped.append({"index": None, "text": text, "token_count": counts[i], "reason": str(e)})
            continue

    return np.array(all_embeddings), all_token_counts, all_skipped

# Save to CSV
def save_embeddings(df, reduced_embeddings, token_counts, filename="embeddings.csv"):
    if len(df) != reduced_embeddings.shape[0]:
        raise ValueError(f"❌ Length mismatch: df has {len(df)} rows, embeddings have {reduced_embeddings.shape[0]}")

    embed_df = pd.DataFrame(reduced_embeddings, columns=[f"dim_{i}" for i in range(reduced_embeddings.shape[1])])
    if "category" in df.columns:
        embed_df["category"] = df["category"].values
    if "cleaned_text" in df.columns:
        embed_df["text"] = df["cleaned_text"].values
    embed_df["token_count"] = token_counts
    
    # embed_df.to_csv(filename, index=False)
    # print(f"✅ Embeddings saved to {filename}")
    return embed_df


# Silhouette Scores and Elbow Score:

def find_optimal_clusters_silhouette(data, max_clusters=10):
    silhouette_scores = [] # create an empty list initially. 
    cluster_range = range(2, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='random', n_init=12, random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score) # append the silhouette scores by adding in "score", which was calculated by the silhoutte_score function. 


    # Optimal k is where the silhouette score is highest
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    return optimal_k

def find_optimal_clusters_elbow(data, max_clusters=10):
    inertia = []
    cluster_range = range(2, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='random', n_init=12, random_state=0) 

        # Number of initializations is 12, random state is just zero. 

        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Find the elbow point using the difference in inertia. Look at the differences

    diff = np.diff(inertia)  # First derivative (Point at which it levels off)
    diff2 = np.diff(diff)  # Second derivative (change in slope)
    optimal_k = cluster_range[np.argmin(diff2) + 1]  # +1 because diff2 is one step shorter

    return optimal_k



