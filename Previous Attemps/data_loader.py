# data_loader.py

from datasets import load_dataset

def load_sample_reviews(sample_size=500):
    print("Starting the data loading process...")

    try:
        print("Attempting to load the Yelp Polarity dataset from Hugging Face...")
        dataset = load_dataset("yelp_polarity")
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Get a sample of reviews
    try:
        print(f"Selecting a sample of {sample_size} reviews from the dataset...")
        reviews_sample = dataset['train'].select(range(sample_size))
        print(f"Sample of {sample_size} reviews selected.")
    except Exception as e:
        print(f"Error selecting sample: {e}")
        return

    # Extract only the review texts for simplicity
    review_texts = reviews_sample['text']
    
    # Save to a file or return the data for further processing
    try:
        print("Saving sample reviews to 'reviews_sample.txt'...")
        with open("reviews_sample.txt", "w", encoding="utf-8") as f:
            for review in review_texts:
                f.write(review + "\n")
        print(f"Sample of {sample_size} reviews successfully saved to 'reviews_sample.txt'")
    except Exception as e:
        print(f"Error saving reviews to file: {e}")

    return review_texts

if __name__ == "__main__":
    print("Executing data_loader.py")
    load_sample_reviews()
    print("Data loading completed successfully!")
