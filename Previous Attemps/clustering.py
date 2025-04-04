import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load embeddings from an .npy file
def load_embeddings(filename="embeddings_v1.npy"):
    """Load embeddings from an .npy file and check if it contains data."""
    try:
        print(f"Loading embeddings from '{filename}'...")
        embeddings = np.load(filename)
        
        if embeddings.size == 0:
            print("Error: Embeddings file is empty.")
            return None
        print(f"Successfully loaded {len(embeddings)} embeddings!")
        return embeddings
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

# Load reviews for direct clustering
def load_reviews(filename="reviews_sample.txt"):
    """Load reviews from a text file."""
    try:
        print(f"Loading reviews from '{filename}'...")
        with open(filename, "r", encoding="utf-8") as f:
            reviews = [line.strip() for line in f.readlines()]
        print(f"Successfully loaded {len(reviews)} reviews!")
        return reviews
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

# Cluster embeddings using KMeans
def cluster_embeddings(embeddings, n_clusters=5):
    """Cluster embeddings using KMeans."""
    if embeddings is None:
        print("No embeddings to cluster. Exiting function.")
        return None

    try:
        print(f"Applying KMeans clustering with {n_clusters} clusters on embeddings...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        print("Clustering complete.")
        return labels
    except Exception as e:
        print(f"Error during clustering: {e}")
        return None

# Cluster reviews directly using TF-IDF and KMeans
def cluster_reviews_tfidf(reviews, n_clusters=5):
    """Cluster raw text reviews using TF-IDF and KMeans."""
    if not reviews:
        print("No reviews to cluster. Exiting function.")
        return None
    
    try:
        print("Transforming reviews into TF-IDF vectors for clustering...")
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(reviews)
        print(f"TF-IDF transformation complete. Matrix shape: {tfidf_matrix.shape}")

        print(f"Applying KMeans clustering with {n_clusters} clusters on TF-IDF vectors...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(tfidf_matrix)
        print("Clustering complete.")
        return labels
    except Exception as e:
        print(f"Error during TF-IDF clustering: {e}")
        return None

# Display clusters of reviews or embeddings
def display_clusters(labels, reviews, n_clusters=5):
    """Display clustered reviews."""
    if labels is None:
        print("No labels to display. Exiting function.")
        return

    print("Displaying clustering results (top 5 reviews per cluster):")
    for i in range(n_clusters):
        print(f"\nCluster {i + 1}")
        cluster_indices = np.where(labels == i)[0]
        for idx in cluster_indices[:5]:  # Display top 5 reviews in each cluster
            print(f"- {reviews[idx]}")

if __name__ == "__main__":
    print("Choose data source for clustering:")
    print("1 - Cluster based on embeddings")
    print("2 - Cluster raw text data")
    choice = input("Enter 1 or 2: ")

    # Set up clustering based on user choice
    if choice == "1":
        # Load and cluster embeddings
        embeddings = load_embeddings()
        if embeddings is not None:
            # Load reviews to display as context for each cluster
            reviews = load_reviews()
            labels = cluster_embeddings(embeddings)
            if labels is not None:
                display_clusters(labels, reviews)
    elif choice == "2":
        # Load and cluster reviews directly using TF-IDF
        reviews = load_reviews()
        if reviews:
            labels = cluster_reviews_tfidf(reviews)
            if labels is not None:
                display_clusters(labels, reviews)
    else:
        print("Invalid choice. Exiting.")
    
    print("Clustering process completed successfully!")

