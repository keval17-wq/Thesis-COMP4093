from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import SUMMARY_OUTPUT_TXT, SUMMARY_OUTPUT_EMB

def load_summaries(filename):
    """Load summaries from a specified text file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            summaries = f.read().split('\n\n')
        summaries = [summary.strip() for summary in summaries if summary.strip()]
        print(f"Loaded {len(summaries)} summaries from {filename}")
        return summaries
    except Exception as e:
        print(f"Error loading summaries: {e}")
        return []

def compute_rouge(reference_summaries, generated_summaries):
    """Compute ROUGE scores between reference and generated summaries."""
    rouge = Rouge()
    scores = []
    for ref, gen in zip(reference_summaries, generated_summaries):
        try:
            score = rouge.get_scores(gen, ref)[0]
            scores.append(score)
        except Exception as e:
            print(f"Error computing ROUGE for a pair of summaries: {e}")
    return scores

def compute_bleu(reference_summaries, generated_summaries):
    """Compute BLEU scores between reference and generated summaries."""
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for ref, gen in zip(reference_summaries, generated_summaries):
        try:
            ref_tokens = ref.split()
            gen_tokens = gen.split()
            score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
            bleu_scores.append(score)
        except Exception as e:
            print(f"Error computing BLEU for a pair of summaries: {e}")
    return bleu_scores

def average_scores(scores):
    """Compute average scores from a list of score dictionaries."""
    avg_scores = {}
    for score in scores:
        for key, value in score.items():
            avg_scores.setdefault(key, 0)
            avg_scores[key] += value
    for key in avg_scores:
        avg_scores[key] /= len(scores)
    return avg_scores

if __name__ == "__main__":
    print("Evaluating summaries generated from both clustering methods.")
    
    # Load summaries generated from embedding-based clustering
    embedding_summaries = load_summaries(SUMMARY_OUTPUT_EMB)
    
    # Load summaries generated from traditional TF-IDF clustering
    traditional_summaries = load_summaries("scenario_3_summaries.txt")

    if len(embedding_summaries) != len(traditional_summaries):
        print("Mismatch in number of summaries for comparison.")
    else:
        # Compute ROUGE scores
        rouge_scores = compute_rouge(traditional_summaries, embedding_summaries)
        avg_rouge = average_scores([score['rouge-l'] for score in rouge_scores])

        # Compute BLEU scores
        bleu_scores = compute_bleu(traditional_summaries, embedding_summaries)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        # Output evaluation results
        print("\nEvaluation Results:")
        print(f"Average ROUGE-L Scores: {avg_rouge}")
        print(f"Average BLEU Score: {avg_bleu:.4f}")
