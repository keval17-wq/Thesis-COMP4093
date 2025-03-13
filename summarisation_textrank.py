#!/usr/bin/env python3
# summarisation_textrank.py

import os
import sys
import openai
import numpy as np
import config
import nltk

from sklearn.cluster import KMeans
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

########################################
# ERROR HANDLING (ENV & OPENMP)
########################################
# Optionally, set this to suppress the OpenMP warning:
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
openai.api_key = config.API_KEY

########################################
# 2. HELPER FUNCTIONS
########################################

def load_reviews(filename=config.DATA_FILE_PATH):
    """Load reviews from the file specified in config.py, with error handling."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reviews = f.readlines()
        reviews = [r.strip() for r in reviews if r.strip()]
        print(f"Loaded {len(reviews)} reviews from {filename}")
        return reviews
    except Exception as e:
        print(f"[Error] Failed to load reviews from {filename}: {e}")
        sys.exit(1)  # Exit on critical error

def generate_review_embeddings(reviews, model=config.MODEL):
    """Generate embeddings for each review (if needed for clustering)."""
    embeddings = []
    for r in reviews:
        try:
            resp = openai.Embedding.create(input=r, model=model)
            emb = resp["data"][0]["embedding"]
        except Exception as e:
            print(f"[Error] Embedding generation failed for review: '{r[:60]}' => {e}")
            emb = [0.0]  # fallback
        embeddings.append(emb)
    return np.array(embeddings)

def run_kmeans(embeddings, num_clusters=3):
    """Perform K-Means clustering with error handling."""
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels
    except Exception as e:
        print(f"[Error] K-Means clustering failed: {e}")
        sys.exit(1)

def summarize_textrank(text, num_sentences=3):
    """Extract key sentences from text using TextRank with sumy."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sents = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary_sents)

def save_summary(summary_text, output_file="summary_textrank.txt"):
    """Save final summary to a file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Summary saved to {output_file}")
    except Exception as e:
        print(f"[Error] Failed to save summary to {output_file}: {e}")

########################################
# 3. MAIN WORKFLOW
########################################
if __name__ == "__main__":
    try:
        # (A) Load Reviews
        reviews = load_reviews()

        # (B) Generate Embeddings (If you want to cluster)
        embeddings = generate_review_embeddings(reviews)

        # (C) Cluster
        labels = run_kmeans(embeddings, num_clusters=3)

        # (D) Summarize Each Cluster via TextRank
        cluster_dict = {}
        for label, review in zip(labels, reviews):
            cluster_dict.setdefault(label, []).append(review)

        full_summary = []
        for cluster_id, cluster_reviews in cluster_dict.items():
            combined_text = " ".join(cluster_reviews)
            # Summarize
            cluster_summary = summarize_textrank(combined_text, num_sentences=2)
            full_summary.append(f"Cluster {cluster_id} => {cluster_summary}")

        final_summary = "\n".join(full_summary)

        # (E) Save summary
        save_summary(final_summary, "summary_textrank.txt")

    except Exception as e:
        print(f"[Error] Unexpected: {e}")
        sys.exit(1)


