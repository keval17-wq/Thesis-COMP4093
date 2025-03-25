#!/usr/bin/env python3
# summarisation_bertsum.py

import os
import sys
from openai import OpenAI

client = OpenAI(api_key=config.API_KEY)
import numpy as np
import config
import nltk

from sklearn.cluster import KMeans
from transformers import pipeline

########################################
# ERROR HANDLING (ENV & OPENMP)
########################################
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer. Please wait...")
    nltk.download("punkt")

########################################
# 1. LOAD API KEY FROM config.py
########################################

########################################
# 2. HELPER FUNCTIONS
########################################

def load_reviews(filename=config.DATA_FILE_PATH):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reviews = f.readlines()
        reviews = [r.strip() for r in reviews if r.strip()]
        print(f"Loaded {len(reviews)} reviews from {filename}")
        return reviews
    except Exception as e:
        print(f"[Error] Failed to load reviews: {e}")
        sys.exit(1)

def generate_review_embeddings(reviews, model=config.MODEL):
    embeddings = []
    for r in reviews:
        try:
            resp = client.embeddings.create(input=r, model=model)
            emb = resp.data[0].embedding
        except Exception as e:
            print(f"[Error] Embedding generation failed for '{r[:60]}' => {e}")
            emb = [0.0]
        embeddings.append(emb)
    return np.array(embeddings)

def run_kmeans(embeddings, num_clusters=3):
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels
    except Exception as e:
        print(f"[Error] K-Means failed: {e}")
        sys.exit(1)

# Initialize BART summarizer (like BERTSUM)
try:
    bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"[Error] Loading BART pipeline failed: {e}")
    bart_summarizer = None

def summarize_bert_bart(text, max_length=80):
    """
    Use BART pipeline to generate a short summary of the combined cluster text.
    This is somewhat extractive in practice, though BART can also be abstractive.
    """
    if not bart_summarizer:
        return "[Error] BART summarizer not available."
    try:
        result = bart_summarizer(text, max_length=max_length, min_length=40, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        print(f"[Error] Summarization with BART failed: {e}")
        return "[Error] Summarization failed."

def save_summary(summary_text, output_file="summary_bert.txt"):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Summary saved to {output_file}")
    except Exception as e:
        print(f"[Error] Failed to save summary to {output_file}: {e}")

########################################
# 3. MAIN PIPELINE
########################################

if __name__ == "__main__":
    try:
        # (A) Load Reviews
        reviews = load_reviews()

        # (B) Generate Embeddings (for clustering)
        embeddings = generate_review_embeddings(reviews)

        # (C) Cluster
        labels = run_kmeans(embeddings, num_clusters=3)

        # (D) Summarize Each Cluster using BERT/BART
        cluster_dict = {}
        for label, review in zip(labels, reviews):
            cluster_dict.setdefault(label, []).append(review)

        full_summary = []
        for cluster_id, cluster_reviews in cluster_dict.items():
            combined_text = " ".join(cluster_reviews)
            # Summarize
            cluster_summary = summarize_bert_bart(combined_text, max_length=80)
            full_summary.append(f"Cluster {cluster_id} => {cluster_summary}")

        final_summary = "\n".join(full_summary)

        # (E) Save summary
        save_summary(final_summary, "summary_bert.txt")

    except Exception as e:
        print(f"[Error] Unexpected: {e}")
        sys.exit(1)
