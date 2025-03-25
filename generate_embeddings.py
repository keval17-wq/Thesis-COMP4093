from openai import OpenAI

client = OpenAI(api_key=API_KEY)
import numpy as np
from config import API_KEY, DATA_FILE_PATH, EMBEDDING_OUTPUT_PATH

# Set the OpenAI API key

def load_reviews(filename=DATA_FILE_PATH):
    """Load reviews from a text file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reviews = f.readlines()
        print(f"Loaded {len(reviews)} reviews from {filename}")
        return [review.strip() for review in reviews]
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

def preprocess_reviews(reviews):
    """Prepare each review by removing extra whitespace and newline characters."""
    processed_reviews = []
    for review in reviews:
        # Remove extra whitespace and ensure single-line format
        processed_review = ' '.join(review.split())
        processed_reviews.append(processed_review)
    print(f"Preprocessed {len(processed_reviews)} reviews.")
    return processed_reviews

def generate_embeddings(reviews):
    """Generate embeddings for each review."""
    embeddings = []
    for review in reviews:
        try:
            response = client.embeddings.create(input=review,
            model="text-embedding-ada-002")
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding for review '{review}': {e}")
    return np.array(embeddings)

def save_embeddings(embeddings, output_path=EMBEDDING_OUTPUT_PATH):
    """Save embeddings to a file in .npy format."""
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    # Step 1: Load reviews
    reviews = load_reviews()

    # Step 2: Preprocess reviews
    if reviews:
        reviews = preprocess_reviews(reviews)

        # Step 3: Generate embeddings if there are reviews
        embeddings = generate_embeddings(reviews)

        # Step 4: Save embeddings
        if embeddings.size > 0:
            save_embeddings(embeddings)
        else:
            print("No embeddings generated.")
    else:
        print("No reviews found to generate embeddings.")

