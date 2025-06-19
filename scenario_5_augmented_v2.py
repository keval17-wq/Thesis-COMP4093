import time, re
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

from config import API_KEY, DATA_FILE_PATH, SUMMARY_OUTPUT_TXT

import openai

client = openai.OpenAI(api_key=API_KEY)

def gpt4_summarize(prompt, instruction):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] GPT summarization failed:", e)
        return None


def load_reviews(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def save_summaries(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def scenario_5_augmented_v2(
    include_sentiment: bool = True,
    top_n: int = 20,
    alpha: float = 0.2,
    num_clusters: int = 10,
    data_path: str = DATA_FILE_PATH,
    output_path: str = SUMMARY_OUTPUT_TXT
):
    """
    Upgraded Scenario 5 pipeline with embedding-based selection, clustering,
    sentiment weighting, and GPT-based summarization.
    """
    start = time.time()
    print(f"\n[DEBUG] Starting Scenario 5 v2")
    print(f"[DEBUG] Reading reviews from: {data_path}")
    print(f"[DEBUG] Sentiment weighting: {include_sentiment}, alpha={alpha}")
    print(f"[DEBUG] Selecting up to {top_n} sentences from {num_clusters} clusters.")

    # 1 · Load and split reviews
    reviews = load_reviews(data_path)
    sentences = [s.strip() for s in re.split(r"[.!?]+", " ".join(reviews)) if s.strip()]
    print(f"[DEBUG] Parsed {len(sentences)} sentences from reviews.")
    if not sentences:
        print("[ERROR] No sentences found – exiting.")
        return

    # 2 · Sentence embeddings
    print(f"[DEBUG] Encoding sentences with MiniLM...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # 3 · Sentiment scoring
    sentiment_scores = [0.0] * len(sentences)
    if include_sentiment:
        print(f"[DEBUG] Calculating sentiment scores...")
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        def get_sentiment(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            logits = sentiment_model(**inputs).logits
            probs = softmax(logits.detach().numpy()[0])
            return probs[2] - probs[0]  # pos - neg

        sentiment_scores = [get_sentiment(s) for s in sentences]
        print(f"[DEBUG] Sentiment scores completed.")

    # 4 · Clustering
    print(f"[DEBUG] Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=min(num_clusters, len(sentences)), random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings.cpu().numpy())

    cluster_sents = {}
    for i, label in enumerate(labels):
        base_score = util.cos_sim(embeddings[i], embeddings.mean(dim=0)).item()
        if include_sentiment:
            base_score *= (1 + alpha * sentiment_scores[i])
        if label not in cluster_sents or cluster_sents[label][1] < base_score:
            cluster_sents[label] = (sentences[i], base_score)

    selected = sorted(cluster_sents.values(), key=lambda x: -x[1])
    top_sentences = [s[0] for s in selected[:top_n]]
    print(f"[DEBUG] Selected {len(top_sentences)} representative sentences.")

    # 5 · GPT summarization
    print(f"[DEBUG] Sending prompt to GPT-4...")
    prompt = (
        "You will receive user-feedback sentences. Compose ONE plain-English "
        "paragraph that is strictly shorter than the combined input, purely factual, "
        "neutral in tone, and mentions each distinct positive and negative fact exactly once. "
        "No bullets or headings.\n\n"
        "Sentences:\n" + "\n".join(f"- {s}" for s in top_sentences)
    )

    summary = gpt4_summarize(prompt, "Return the requested paragraph only.") or "Summary generation failed."
    source_len = len("\n".join(top_sentences))

    if len(summary) >= source_len:
        print(f"[DEBUG] Summary too long, compressing...")
        compress_prompt = (
            "Rewrite the following paragraph so it remains purely factual but is "
            "STRICTLY SHORTER in character count than the original source sentences. "
            "Keep all distinct facts.\n\n" + summary
        )
        summary = gpt4_summarize(compress_prompt, "Return the shorter paragraph only.") or summary

    if len(summary) >= source_len:
        print(f"[DEBUG] Summary still too long. Truncating manually.")
        summary = summary[:source_len - 1].rsplit(" ", 1)[0].rstrip(",.;: ") + "…"

    # 6 · Save results
    print(f"[DEBUG] Saving results to: {output_path}")
    save_summaries(
        [
            summary
        ],
        output_path
    )

    print(f"[DEBUG] Scenario 5 v2 complete. Time: {time.time() - start:.2f}s\n")

if __name__ == "__main__":
    scenario_5_augmented_v2()
