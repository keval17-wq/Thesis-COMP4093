import sys
import re
import time
import numpy as np
from openai import OpenAI

# Your config file references
from config import API_KEY, DATA_FILE_PATH, SUMMARY_OUTPUT_TXT, SUMMARY_OUTPUT_EMB

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Attempt to import Hugging Face pipeline for optional sentiment
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
except ImportError:
    sentiment_pipeline = None
    print("Install `transformers` if you want to run optional sentiment analysis in scenario_5.")

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# Common Helpers
# -------------------------------
def load_reviews(filename):
    """Load lines from a file, ignoring empty lines."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        reviews = [line.strip() for line in lines if line.strip()]
        print(f"Loaded {len(reviews)} reviews from {filename}")
        return reviews
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

def gpt4_summarize(text, system_instruction="You are an expert summarizer focused on conciseness and clarity."):
    """Use GPT-4 to summarize text with a given system instruction."""
    if not text.strip():
        return "No meaningful text to summarize."
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def save_summaries(summaries, output_path):
    """Save multiple summary strings to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for s in summaries:
                f.write(s + "\n\n")
        print(f"Saved summaries to {output_path}")
    except Exception as e:
        print(f"Error saving summaries: {e}")


# -------------------------------
# Scenario 1
# -------------------------------
def generate_summaries(reviews):
    """Generate GPT-4 summaries for each review in the list (one by one)."""
    results = []
    for rev in reviews:
        prompt = f"Please provide a concise summary of the following text:\n\n{rev}"
        summary = gpt4_summarize(prompt)
        if summary:
            results.append(summary)
            print(f"Original (first 60 chars): {rev[:60]}... | Summary (first 60 chars): {summary[:60]}...")
        else:
            results.append("Summary generation failed.")
    return results

def scenario_1():
    """Summarize first 15 raw reviews individually."""
    start_time = time.time()
    reviews = load_reviews(DATA_FILE_PATH)[:15]
    summaries = generate_summaries(reviews)
    save_summaries(summaries, SUMMARY_OUTPUT_TXT)
    elapsed = time.time() - start_time
    print(f"Scenario 1 completed in {elapsed:.2f} seconds.")


# -------------------------------
# Scenario 2
# -------------------------------
def generate_embeddings(reviews):
    """Generate embeddings (text-embedding-ada-002) for a list of reviews."""
    embeddings = []
    for r in reviews:
        try:
            resp = client.embeddings.create(input=r, model="text-embedding-ada-002")
            embeddings.append(resp.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding for '{r[:60]}': {e}")
            embeddings.append([0.0]*1536)
    return np.array(embeddings)

def cluster_reviews(embeddings, reviews, n_clusters=5):
    """Cluster reviews by embeddings using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {}
    for label, rev in zip(labels, reviews):
        clusters.setdefault(label, []).append(rev)
    print(f"Reviews clustered into {n_clusters} clusters.")
    return clusters

def scenario_2():
    """Embed reviews -> K-Means -> Summarize each cluster with GPT-4."""
    start_time = time.time()
    reviews = load_reviews(DATA_FILE_PATH)
    emb = generate_embeddings(reviews)
    clusters_dict = cluster_reviews(emb, reviews, n_clusters=5)

    summaries = []
    for cid, revs_in_cluster in clusters_dict.items():
        cluster_text = " ".join(revs_in_cluster)
        prompt = f"Please summarize this cluster of user feedback:\n\n{cluster_text}"
        s = gpt4_summarize(prompt, "You are an expert summarizer focusing on the main themes.")
        summaries.append(f"Cluster {cid}:\n{s if s else 'Summary generation failed.'}")

    save_summaries(summaries, SUMMARY_OUTPUT_EMB)
    elapsed = time.time() - start_time
    print(f"Scenario 2 completed in {elapsed:.2f} seconds.")


# -------------------------------
# Scenario 3
# -------------------------------
def traditional_cluster_reviews(reviews, n_clusters=5):
    """Cluster reviews using TF-IDF + K-Means."""
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(reviews)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    clusters = {}
    for label, rev in zip(labels, reviews):
        clusters.setdefault(label, []).append(rev)
    print(f"Reviews clustered into {n_clusters} clusters using TF-IDF.")
    return clusters

def scenario_3():
    """TF-IDF + K-Means -> Summarize each cluster with GPT-4."""
    start_time = time.time()
    reviews = load_reviews(DATA_FILE_PATH)
    clusters = traditional_cluster_reviews(reviews, n_clusters=5)

    results = []
    for cid, revs in clusters.items():
        text_block = " ".join(revs)
        prompt = f"Please summarize this cluster of user feedback:\n\n{text_block}"
        s = gpt4_summarize(prompt, "You are an expert summarizer focusing on the main themes.")
        results.append(f"Cluster {cid}:\n{s if s else 'Summary generation failed.'}")

    save_summaries(results, 'scenario_3_summaries.txt')
    elapsed = time.time() - start_time
    print(f"Scenario 3 completed in {elapsed:.2f} seconds.")


# -------------------------------
# Scenario 4
# -------------------------------
def scenario_4():
    """
    TF-IDF-based extractive step + final GPT-4 summarization.
    """
    start_time = time.time()
    reviews = load_reviews(DATA_FILE_PATH)
    all_text = " ".join(reviews)
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        print("No sentences to process.")
        return

    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    sorted_idx = np.argsort(scores)[::-1]

    top_n = min(5, len(sorted_idx))
    top_sents = [sentences[i] for i in sorted_idx[:top_n]]
    extractive_draft = ". ".join(top_sents) + "."

    prompt = (
        f"These are the top sentences I extracted:\n\n{extractive_draft}\n\n"
        "Please generate a single-paragraph summary that captures all main themes, issues, and sentiments."
    )
    final_sum = gpt4_summarize(prompt, "You are an expert summarizer ensuring completeness and coherence.")
    if not final_sum:
        final_sum = "Summary generation failed."

    res = [
        "Extractive Draft Sentences:\n" + extractive_draft,
        "Final Abstractive Summary:\n" + final_sum
    ]
    save_summaries(res, "scenario_4_summary.txt")
    elapsed = time.time() - start_time
    print(f"Scenario 4 completed in {elapsed:.2f} seconds.")


# -------------------------------
# Scenario 5 (with debugging, optional sentiment)
# -------------------------------
def scenario_5(include_sentiment=True, top_n=15):
    """
    Novelty/Hybrid approach:
      1) Use TF-IDF to extract top N sentences from all reviews.
      2) (Optional) run sentiment analysis on each extracted sentence.
      3) Summarize the result using GPT-4.
    """
    import time
    start_time = time.time()
    print(f"[DEBUG] Starting Scenario 5 (include_sentiment={include_sentiment}, top_n={top_n})")

    # 1) Load and extract
    reviews = load_reviews(DATA_FILE_PATH)
    full_text = " ".join(reviews)
    sentences = re.split(r'[.!?]+', full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"[DEBUG] Found {len(sentences)} total sentences.")

    if not sentences:
        print("[DEBUG] No sentences found. Exiting Scenario 5.")
        return

    # TF-IDF extraction
    print("[DEBUG] Performing TF-IDF scoring.")
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    sorted_indices = np.argsort(scores)[::-1]

    top_n = min(top_n, len(sorted_indices))
    top_sents = [sentences[i] for i in sorted_indices[:top_n]]
    print(f"[DEBUG] Extracted top {top_n} sentences.")

    # 2) Optional sentiment
    labeled_sents = []
    if include_sentiment and sentiment_pipeline:
        print("[DEBUG] Running sentiment analysis.")
        results = sentiment_pipeline(top_sents)
        for s, r in zip(top_sents, results):
            labeled_sents.append(f"• {s}\n   Sentiment: {r['label']} (score={r['score']:.2f})")
    else:
        if include_sentiment and not sentiment_pipeline:
            print("[DEBUG] Sentiment pipeline not available. Skipping sentiment.")
        labeled_sents = [f"• {s}" for s in top_sents]

    # 3) Final GPT summary
    print("[DEBUG] Calling GPT-4 for final summary.")
    prompt = (
        "Below are key extracted sentences from user feedback.\n"
        "Some include sentiment labels:\n\n"
        + "\n\n".join(labeled_sents)
        + "\n\nPlease generate a concise, high-level summary of major themes and sentiments and categorise them by Product teams: Summarize feature requests and technical issues. Customer support: Emphasize complaints and service-related feedback. Executives: Provide a high-level overview of major trends and insights."
    )
    final_summary = gpt4_summarize(prompt, "You are an expert summarizer identifying insights from feedback.")

    if not final_summary:
        final_summary = "Summary generation failed."

    # Save
    output_data = [
        "Extracted Sentences:\n" + "\n".join(top_sents),
        "\nSentiment-Labeled Sentences:\n" + "\n".join(labeled_sents),
        "\nFinal GPT Summary:\n" + final_summary
    ]
    save_summaries(output_data, "scenario_5_summary.txt")

    elapsed = time.time() - start_time
    print(f"[DEBUG] Scenario 5 completed in {elapsed:.2f} seconds.")
    print("End of program.")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("Select scenario to run:")
    print("1 - Summarize first 15 raw reviews (individual GPT-4 summaries)")
    print("2 - Embed + K-Means + Summarize clusters (GPT-4)")
    print("3 - TF-IDF + K-Means + Summarize clusters (GPT-4)")
    print("4 - Extractive (TF–IDF) + Abstractive (GPT-4) hybrid summary (entire dataset)")
    print("5 - TF–IDF extractive + (Optional) Sentiment + GPT-4 final summary")
    choice = input("Enter 1, 2, 3, 4, or 5: ")

    if choice == "1":
        scenario_1()
    elif choice == "2":
        scenario_2()
    elif choice == "3":
        scenario_3()
    elif choice == "4":
        scenario_4()
    elif choice == "5":
        # Toggle sentiment if you want
        scenario_5(include_sentiment=True)
    else:
        print("Invalid choice. Exiting.")
        sys.exit()
