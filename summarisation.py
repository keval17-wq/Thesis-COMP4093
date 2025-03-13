import sys
import re
import numpy as np
import openai
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Import config variables (same as before)
from config import API_KEY, DATA_FILE_PATH, SUMMARY_OUTPUT_TXT, SUMMARY_OUTPUT_EMB

# Set OpenAI API key from config
openai.api_key = API_KEY

def load_reviews(filename):
    """Load reviews from a specified text file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reviews = f.readlines()
        print(f"Loaded {len(reviews)} reviews from {filename}")
        # Filter out empty lines
        return [review.strip() for review in reviews if review.strip()]
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

def gpt4_summarize(text, system_instruction="You are an expert summarizer focused on conciseness and clarity."):
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

def generate_summaries(reviews):
    """Generate GPT-4 summaries for each review in the list."""
    summaries = []
    for review in reviews:
        prompt = f"Please provide a concise summary of the following text:\n\n{review}"
        summary = gpt4_summarize(prompt)
        if summary:
            summaries.append(summary)
            print(f"Original: {review[:60]}... | Summary: {summary[:60]}...")
        else:
            summaries.append("Summary generation failed.")
    return summaries

def generate_embeddings(reviews):
    """Generate embeddings for a list of reviews using OpenAI embeddings."""
    embeddings = []
    for review in reviews:
        try:
            response = openai.Embedding.create(
                input=review,
                model="text-embedding-ada-002"
            )
            embeddings.append(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error generating embedding for review '{review[:60]}': {e}")
            # Fallback to a zero embedding to maintain shape
            embeddings.append([0.0]*1536)
    return np.array(embeddings)

def cluster_reviews(embeddings, reviews, n_clusters=5):
    """Cluster reviews based on embeddings using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {}
    for label, review in zip(labels, reviews):
        clusters.setdefault(label, []).append(review)
    print(f"Reviews clustered into {n_clusters} clusters.")
    return clusters

def traditional_cluster_reviews(reviews, n_clusters=5):
    """Cluster reviews without embeddings using TF-IDF and K-Means."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    clusters = {}
    for label, review in zip(labels, reviews):
        clusters.setdefault(label, []).append(review)
    print(f"Reviews clustered into {n_clusters} clusters using traditional methods.")
    return clusters


# -------------------------------
# Scenario 1
# -------------------------------
def scenario_1():
    """Scenario 1: Summarize the first 15 raw reviews individually."""
    reviews = load_reviews(DATA_FILE_PATH)
    reviews = reviews[:15]
    summaries = generate_summaries(reviews)
    save_summaries(summaries, SUMMARY_OUTPUT_TXT)


# -------------------------------
# Scenario 2
# -------------------------------
def scenario_2():
    """Scenario 2: Embed reviews, cluster them, then summarize each cluster as a whole."""
    reviews = load_reviews(DATA_FILE_PATH)
    embeddings = generate_embeddings(reviews)
    clusters_dict = cluster_reviews(embeddings, reviews, n_clusters=5)
    
    summaries = []
    for cluster_id, reviews_in_cluster in clusters_dict.items():
        cluster_text = " ".join(reviews_in_cluster)
        
        # Summarize cluster text with GPT-4
        prompt = f"Please summarize this cluster of user feedback:\n\n{cluster_text}"
        summary = gpt4_summarize(prompt, "You are an expert summarizer focusing on the main themes.")
        
        if summary:
            summaries.append(f"Cluster {cluster_id}:\n{summary}")
        else:
            summaries.append(f"Cluster {cluster_id}:\nSummary generation failed.")
    save_summaries(summaries, SUMMARY_OUTPUT_EMB)


# -------------------------------
# Scenario 3
# -------------------------------
def scenario_3():
    """Scenario 3: Cluster reviews using TF-IDF (no embeddings), then summarize each cluster."""
    reviews = load_reviews(DATA_FILE_PATH)
    clusters = traditional_cluster_reviews(reviews, n_clusters=5)
    
    summaries = []
    for cluster_id, cluster_reviews in clusters.items():
        cluster_text = " ".join(cluster_reviews)
        
        # Summarize cluster text with GPT-4
        prompt = f"Please summarize this cluster of user feedback:\n\n{cluster_text}"
        summary = gpt4_summarize(prompt, "You are an expert summarizer focusing on the main themes.")
        
        if summary:
            summaries.append(f"Cluster {cluster_id}:\n{summary}")
        else:
            summaries.append(f"Cluster {cluster_id}:\nSummary generation failed.")

    save_summaries(summaries, 'scenario_3_summaries.txt')


# -------------------------------
# Scenario 4 (NEW)
# -------------------------------
def scenario_4():
    """
    Scenario 4: 
    - Perform a simple extractive step by picking top sentences via TF–IDF
      from the entire set of reviews.
    - Then feed these extracted sentences to GPT-4 for a final single-paragraph summary
      capturing major themes.
    """
    reviews = load_reviews(DATA_FILE_PATH)

    # Combine all reviews into one big text block
    all_text = " ".join(reviews)

    # Naive sentence splitting (avoid extra libs)
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        print("No sentences to process.")
        return

    # Use TF-IDF at the sentence level
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    # Score each sentence by sum of TF-IDF values
    scores = X.sum(axis=1).A1
    # Sort sentences by descending score
    sorted_indices = np.argsort(scores)[::-1]

    # Pick top N sentences
    top_n = min(5, len(sorted_indices))
    top_sentences = [sentences[i] for i in sorted_indices[:top_n]]

    # Create an extractive draft
    extractive_draft = ". ".join(top_sentences) + "."

    # Abstractive step with GPT-4
    prompt = (
        f"These are the top sentences I extracted:\n\n{extractive_draft}\n\n"
        "Please generate a single-paragraph summary that captures all main themes, issues, and sentiments "
        "in a concise manner."
    )
    final_summary = gpt4_summarize(prompt, "You are an expert summarizer ensuring completeness and coherence.")

    if not final_summary:
        final_summary = "Summary generation failed."

    # Save the final result
    result = [
        "Extractive Draft Sentences:\n" + extractive_draft,
        "Final Abstractive Summary:\n" + final_summary
    ]
    save_summaries(result, "scenario_4_summary.txt")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("Select scenario to run:")
    print("1 - Summarize first 15 raw reviews (individual GPT-4 summaries)")
    print("2 - Embed + K-Means + Summarize clusters (GPT-4)")
    print("3 - TF-IDF + K-Means + Summarize clusters (GPT-4)")
    print("4 - Extractive (TF–IDF) + Abstractive (GPT-4) hybrid summary (entire dataset)")
    choice = input("Enter 1, 2, 3, or 4: ")

    if choice == "1":
        scenario_1()
    elif choice == "2":
        scenario_2()
    elif choice == "3":
        scenario_3()
    elif choice == "4":
        scenario_4()
    else:
        print("Invalid choice. Exiting.")
        sys.exit()


# import openai
# from config import API_KEY, DATA_FILE_PATH, SUMMARY_OUTPUT_TXT, SUMMARY_OUTPUT_EMB
# import sys
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Set OpenAI API key from config
# openai.api_key = API_KEY

# def load_reviews(filename):
#     """Load reviews from a specified text file."""
#     try:
#         with open(filename, "r", encoding="utf-8") as f:
#             reviews = f.readlines()
#         print(f"Loaded {len(reviews)} reviews from {filename}")
#         return [review.strip() for review in reviews]
#     except Exception as e:
#         print(f"Error loading reviews: {e}")
#         return []

# def summarize_text(text):
#     """Summarize a single text entry using OpenAI's GPT-4 model."""
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an expert summarizer focused on conciseness and clarity."},
#                 {"role": "user", "content": f"Please provide a concise summary of the following text:\n\n{text}"}
#             ]
#         )
#         summary = response['choices'][0]['message']['content'].strip()
#         return summary
#     except Exception as e:
#         print(f"Error summarizing text: {e}")
#         return None

# def generate_summaries(reviews):
#     """Generate summaries for each review in the list."""
#     summaries = []
#     for review in reviews:
#         summary = summarize_text(review)
#         if summary:
#             summaries.append(summary)
#             print(f"Original: {review[:60]}... | Summary: {summary[:60]}...")
#         else:
#             summaries.append("Summary generation failed.")
#     return summaries

# def save_summaries(summaries, output_path):
#     """Save summaries to a specified file."""
#     try:
#         with open(output_path, "w", encoding="utf-8") as f:
#             for summary in summaries:
#                 f.write(summary + "\n\n")
#         print(f"Saved summaries to {output_path}")
#     except Exception as e:
#         print(f"Error saving summaries: {e}")

# def generate_embeddings(reviews):
#     """Generate embeddings for a list of reviews using OpenAI embeddings."""
#     embeddings = []
#     for review in reviews:
#         try:
#             response = openai.Embedding.create(
#                 input=review,
#                 model="text-embedding-ada-002"
#             )
#             embeddings.append(response['data'][0]['embedding'])
#         except Exception as e:
#             print(f"Error generating embedding for review '{review}': {e}")
#     return np.array(embeddings)

# def cluster_reviews(embeddings, reviews, n_clusters=5):
#     """Cluster reviews based on embeddings using K-Means."""
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(embeddings)
#     clusters = {}
#     for label, review in zip(labels, reviews):
#         clusters.setdefault(label, []).append(review)
#     print(f"Reviews clustered into {n_clusters} clusters.")
#     return clusters

# def traditional_cluster_reviews(reviews, n_clusters=5):
#     """Cluster reviews without embeddings using TF-IDF and K-Means."""
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(reviews)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(X)
#     clusters = {}
#     for label, review in zip(labels, reviews):
#         clusters.setdefault(label, []).append(review)
#     print(f"Reviews clustered into {n_clusters} clusters using traditional methods.")
#     return clusters

# def scenario_1():
#     """Scenario 1: Summarize the first 15 raw reviews."""
#     reviews = load_reviews(DATA_FILE_PATH)
#     reviews = reviews[:15]
#     summaries = generate_summaries(reviews)
#     save_summaries(summaries, SUMMARY_OUTPUT_TXT)

# def scenario_2():
#     """Scenario 2: Generate embeddings, cluster reviews, then summarize each cluster as a whole with themes."""
#     # Step 1: Load reviews
#     reviews = load_reviews(DATA_FILE_PATH)
    
#     # Step 2: Generate embeddings for the reviews
#     embeddings = generate_embeddings(reviews)
    
#     # Step 3: Cluster the reviews based on embeddings
#     clusters_dict = cluster_reviews(embeddings, reviews)  # Function returns clusters
    
#     summaries = []
    
#     # Step 4: Generate a theme and summary for each cluster
#     for cluster_id, reviews_in_cluster in clusters_dict.items():
#         # Combine all reviews in the cluster into a single text block
#         cluster_text = " ".join(reviews_in_cluster)
        
#         # Generate a theme for the cluster
#         theme = summarize_text(f"Identify the key issues or main concerns in the following text and keep it very succint, no sentences, straight to point themes: \n\n{cluster_text}")
        
#         # Generate a summary for the entire cluster
#         summary = summarize_text(cluster_text)
        
#         # Store the theme and summary with labels for the cluster
#         if theme and summary:
#             summaries.append(f"Cluster {cluster_id} Theme: {theme}\nSummary:\n{summary}\n\n")
#         elif summary:  # If theme generation fails but summary is available
#             summaries.append(f"Cluster {cluster_id} Summary:\n{summary}\n\n")
#         else:
#             summaries.append(f"Cluster {cluster_id} Summary:\nSummary generation failed.\n\n")

#     # Step 5: Save all themes and summaries to the output file
#     save_summaries(summaries, SUMMARY_OUTPUT_EMB)


# def scenario_3():
#     """Scenario 3: Cluster without embeddings, then summarize each cluster as a whole."""
#     reviews = load_reviews(DATA_FILE_PATH)
#     clusters = traditional_cluster_reviews(reviews)
#     summaries = []
    
#     # Summarize each cluster as a whole
#     for cluster_id, cluster_reviews in clusters.items():
#         # Combine all reviews in the cluster into a single text
#         cluster_text = " ".join(cluster_reviews)
#         # Summarize the entire cluster
#         summary = summarize_text(cluster_text)
#         if summary:
#             summaries.append(f"Cluster {cluster_id} Summary:\n{summary}\n\n")
#         else:
#             summaries.append(f"Cluster {cluster_id} Summary:\nSummary generation failed.\n\n")

#     save_summaries(summaries, 'scenario_3_summaries.txt')

# if __name__ == "__main__":
#     print("Select scenario to run:")
#     print("1 - Summarize first 15 raw reviews")
#     print("2 - Embed, cluster, then summarize")
#     print("3 - Cluster without embeddings, then summarize")
#     choice = input("Enter 1, 2, or 3: ")

#     if choice == "1":
#         scenario_1()
#     elif choice == "2":
#         scenario_2()
#     elif choice == "3":
#         scenario_3()
#     else:
#         print("Invalid choice. Exiting.")
#         sys.exit()


