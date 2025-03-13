"""
Summarizer Code with a Novel Extractive-Abstractive Approach (No Clustering)

Requirements:
- openai (for GPT-4 usage)
- transformers (for Hugging Face summarization pipeline)
- textblob (for sentiment analysis)
- scikit-learn, numpy, etc. (already used in previous code)

Instructions:
1. Install all necessary packages:
   pip install openai transformers textblob scikit-learn numpy

2. Prepare a config.py file with the following variables:
   - API_KEY = "YOUR_OPENAI_API_KEY"
   - DATA_FILE_PATH = "path/to/your/reviews.txt"
   - SUMMARY_OUTPUT_TXT = "output_summary.txt"

3. Run this script:
   python summarizer_novel_approach.py
"""

import sys
import re
import time
import numpy as np
import openai

# Sentiment analysis
from textblob import TextBlob

# For Hugging Face extraction
from transformers import pipeline

from config import API_KEY, DATA_FILE_PATH, SUMMARY_OUTPUT_TXT

###############################
# OpenAI and HF Configuration #
###############################
openai.api_key = API_KEY

# Create an extractive summarization pipeline (Hugging Face)
extractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

###############################
# Utility Functions           #
###############################
def load_reviews(filename):
    """Load reviews from a specified text file and return them as a list of strings."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reviews = f.readlines()
        # Filter out empty lines and strip whitespace
        reviews = [r.strip() for r in reviews if r.strip()]
        print(f"Loaded {len(reviews)} reviews from {filename}")
        return reviews
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []


def gpt4_summarize(text, system_instruction="You are an expert summarizer focused on clarity."):
    """Use GPT-4 to summarize text with a given system instruction."""
    if not text.strip():
        return "No meaningful text to summarize."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
            ]
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None


def save_summaries(summaries, output_path):
    """Save summaries to a specified file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for summary in summaries:
                f.write(summary + "\n\n")
        print(f"Saved summaries to {output_path}")
    except Exception as e:
        print(f"Error saving summaries: {e}")

###############################
# Novel Extractive-Abstractive Approach
###############################

def multi_pass_extract(text):
    """
    Multi-pass extraction using a Hugging Face summarization pipeline.
    1. Summarize entire text to get a condensed version.
    2. Split that summary into sentences.
    3. (Optional) Additional pass can refine or remove redundancies.
    Returns a list of candidate sentences.
    """
    if not text.strip():
        return []

    # First pass: Hugging Face extractive summarization
    # Adjust max_length/min_length as needed
    extracted = extractive_summarizer(text, max_length=200, min_length=30, do_sample=False)
    # Typically returns a list of dicts with 'summary_text'
    if not extracted:
        return []

    summary_text = extracted[0]['summary_text']

    # Split into sentences by punctuation
    sentences = re.split(r'[.!?]+', summary_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Additional pass for very large texts (if needed) can be added here.

    return sentences


def score_sentences(sentences):
    """
    Score sentences based on novelty (unique words) and sentiment magnitude.
    Returns a sorted list of sentences (descending by score).
    """
    if not sentences:
        return []

    seen_phrases = set()
    scored = []

    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity  # -1.0 (negative) to 1.0 (positive)
        words = sentence.lower().split()

        # novelty = number of new words not seen before
        novelty = sum(1 for w in words if w not in seen_phrases)

        # update seen phrases with current sentence's words
        seen_phrases.update(words)

        # Weighted score: novelty plus some weighting for sentiment
        score = novelty + abs(polarity * 2)
        scored.append((sentence, score))

    # sort by score descending
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    # return only sentences
    return [s[0] for s in scored_sorted]


def query_aware_summary(sentences, query="general user insights"):
    """
    Use GPT-4 to create an abstractive summary, focusing on a particular query.
    """
    if not sentences:
        return "No sentences provided."

    combined = ". ".join(sentences) + "."
    prompt = (
        f"Here are key points from user feedback:\n\n{combined}\n\n"
        f"Please summarize them with a focus on: {query}. Provide a concise, structured summary."
    )

    result = gpt4_summarize(prompt, system_instruction="You are an expert summarizer ensuring completeness.")
    return result if result else "Summary generation failed."

###############################
# Main Scenario
###############################
def scenario_novel():
    """
    Scenario: Multi-pass extractive (Hugging Face) + sentiment-based ranking + query-aware abstractive summarization.
    """
    start_time = time.time()

    # 1. Load reviews
    reviews = load_reviews(DATA_FILE_PATH)
    if not reviews:
        print("No reviews loaded. Exiting.")
        return

    # 2. Combine all reviews into one text block
    all_text = " ".join(reviews)

    # 3. Multi-pass extraction
    extracted_sentences = multi_pass_extract(all_text)
    if not extracted_sentences:
        print("No sentences extracted. Exiting.")
        return

    # 4. Score sentences (novelty + sentiment)
    scored_sentences = score_sentences(extracted_sentences)

    # 5. Query-aware summary using GPT-4
    # You can change this query to reflect your exact needs.
    query = "key user concerns and positive highlights"
    final_summary = query_aware_summary(scored_sentences, query)

    # 6. Save results
    results = [
        "Extracted Sentences (sorted by score):\n" + "\n".join(scored_sentences),
        "\nFinal Abstractive Summary:\n" + final_summary
    ]
    save_summaries(results, SUMMARY_OUTPUT_TXT)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Scenario complete. Time taken: {elapsed:.2f} seconds.")

###############################
# Main Entry
###############################
if __name__ == "__main__":
    # Simply run the scenario
    scenario_novel()
