import sys
import os

# Add src directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from collections import defaultdict
import math
from utils import load_data, split_by_language

def train_bigram_model(names):
    """
    Trains a character-level bigram model.
    Returns:
        prob_matrix: A dictionary of dictionaries where prob_matrix[char1][char2] = P(char2 | char1)
    """
    bigram_counts = defaultdict(lambda: defaultdict(int))
    unigram_counts = defaultdict(int)
    
    # Add start and end tokens
    processed_names = ["^" + name.lower() + "$" for name in names]
    
    for name in processed_names:
        for i in range(len(name) - 1):
            char1 = name[i]
            char2 = name[i+1]
            bigram_counts[char1][char2] += 1
            unigram_counts[char1] += 1
            
    # Convert counts to probabilities logic
    # P(char2 | char1) = count(char1, char2) / count(char1)
    prob_matrix = defaultdict(dict)
    
    for char1, following_chars in bigram_counts.items():
        total_count = unigram_counts[char1]
        for char2, count in following_chars.items():
            prob_matrix[char1][char2] = count / total_count
            
    return prob_matrix

def calculate_likelihood(model, name):
    """
    Calculates the likelihood and log-likelihood of a name using the bigram model.
    """
    name_processed = "^" + name.lower() + "$"
    log_likelihood = 0
    probability = 1.0
    
    path_details = []
    
    for i in range(len(name_processed) - 1):
        char1 = name_processed[i]
        char2 = name_processed[i+1]
        
        if char1 in model and char2 in model[char1]:
            prob = model[char1][char2]
        else:
            # Handle unseen bigrams with a very small probability (smoothing could be used here, but for now strict)
            prob = 1e-10 
            
        probability *= prob
        log_likelihood += math.log(prob)
        path_details.append(f"P({char2}|{char1})={prob:.4f}")
        
    return probability, log_likelihood, path_details

def generate_completion(model, prefix):
    """
    Completes a name given a prefix using the most likely next character.
    """
    current_name = prefix
    current_char = prefix[-1].lower() # Assume prefix is at least length 1
    
    # Check if the prefix itself is valid (should start with start token logic implicitly)
    # But here we just continue from the last character
    
    max_len = 20
    while len(current_name) < max_len:
        if current_char not in model:
            break
            
        # Greedily choose the most likely next character
        next_chars_probs = model[current_char]
        if not next_chars_probs:
            break
            
        best_next_char = max(next_chars_probs, key=next_chars_probs.get)
        
        if best_next_char == '$':
            break
            
        current_name += best_next_char
        current_char = best_next_char
        
    return current_name

def main():
    names, labels = load_data()
    english_names, _ = split_by_language(names, labels)
    
    print(f"Training on {len(english_names)} English names...")
    model = train_bigram_model(english_names)
    
    # Part a) Likelihoods
    test_names = [
        "Fergus", "Angus", "Boston", "Austin", "Dankworth", 
        "Denkworth", "Birtwistle", "Birdwhistle"
    ]
    
    print("\n--- Part a) Likelihoods ---")
    print(f"{'Name':<15} {'Log-Likelihood':<20} {'Probability':<20}")
    print("-" * 55)
    
    results = []
    for name in test_names:
        prob, log_prob, _ = calculate_likelihood(model, name)
        results.append((name, prob, log_prob))
        print(f"{name:<15} {log_prob:<20.4f} {prob:.4e}")
        
    # Part b) Completions
    prefixes = ["Lou", "Ber", "Cul", "Ede", "Zjo"]
    print("\n--- Part b) Most Likely Completions ---")
    for prefix in prefixes:
        completion = generate_completion(model, prefix)
        print(f"Prefix: {prefix:<5} -> Completion: {completion}")

    # Part c) Critique (printed for now, to be included in report)
    print("\n--- Part c) Critique ---")
    print("One least available result from a) or b) and how to improve.")
    # Automated selection of least plausible
    # Sort by log-likelihood
    results.sort(key=lambda x: x[2])
    worst_name = results[0]
    print(f"Lowest likelihood name: {worst_name[0]} with log-prob {worst_name[2]:.4f}")

if __name__ == "__main__":
    main()
