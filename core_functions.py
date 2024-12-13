import numpy as np
import config
import re
import itertools
import functions as f
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import imblearn.over_sampling as sm
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

def preprocess(data):
    replacements = config.replacements
    rename_map = config.feature_rename_map
    known_brands = config.brand_set

    model_ids = f.extract_values(data, 'modelID')
    shops = f.extract_values(data, 'shop')
    feature_maps = f.extract_values(data, 'featuresMap')
    titles = f.extract_values(data, 'title')
    urls = f.extract_values(data, 'url')

    titles_clean = [f.clean_text(title, replacements) for title in titles]
    
    brands_cleaned = [f.extract_brand_from_title(title, known_brands) for title in titles_clean]

    url_clean = [f.clean_text(url, replacements) for url in urls]

    feature_maps_clean = [
        {f.clean_text(key, replacements): f.clean_text(value, replacements) for key, value in feature_map.items()}
        for feature_map in feature_maps]

    # Apply the renaming to the cleaned feature maps
    feature_maps_renamed = []
    for feature_map in feature_maps_clean:
        renamed_map = {}
        for feature, value in feature_map.items():
            new_feature = f.rename_feature(feature, rename_map)  # Use the rename function
            renamed_map[new_feature] = value  # Retain the value for the renamed feature
        feature_maps_renamed.append(renamed_map)  # Collect the renamed map
        
    screen_sizes = f.extract_quantity(titles_clean, 'inch', 2)
    refresh_rates = f.extract_quantity(titles_clean, 'hz', 3)

    return {
        "model_ids_clean": model_ids,
        "shops_clean": shops,
        "feature_maps_renamed": feature_maps_renamed,
        "titles_clean": titles_clean,
        "urls": url_clean,
        "brands_cleaned": brands_cleaned,
        "screens": screen_sizes,
        "refresh_rates": refresh_rates
    }
    

def lsh_par(model_ids, titles, shops,  k_shingle_length, hashes, bands, lsh=True):
    #Pair Filtering
    n_items = len(titles)
    shop_match = f.pair_match(shops)
    candidate_pairs = [(i, j) for i in range(n_items - 1) for j in range(i + 1, n_items) if not shop_match[i][j]]
    tot_comparisons = len(candidate_pairs)

    if lsh:
        titles_shingled = f.shingle_matrix(titles, k_shingle_length)
        sig_matrix = f.min_hash(titles_shingled, hashes)
        candidate_pairs = f.lsh(sig_matrix, bands)
        candidate_pairs = [(i, j) for (i, j) in candidate_pairs if not shop_match[i][j]]

    frac_comp = len(candidate_pairs) / tot_comparisons
    actual_pairs = [(i, j) for i in range(n_items-1) for j in range(i+1, n_items) if model_ids[i] == model_ids[j]]
    pair_quality_measures = f.accuracy_measures(candidate_pairs, actual_pairs)

    return candidate_pairs, actual_pairs, frac_comp, pair_quality_measures


def combine_features_with_title(titles, feature_values_list):
    # Iterate through the titles and their corresponding feature value lists
    for i in range(len(titles)):
        title = titles[i]
        feature_values = feature_values_list[i]
        
        # Split the title into words
        title_words = set(title.split())
        
        # Add words from feature_values to the title if they're not already present
        for feature in feature_values:
            if feature not in title_words:
                title += " " + feature  # Add feature to title if not present

        # Update the title in the list
        titles[i] = title

    return titles

def map_clusters_and_generate_pairs(unique_list, cluster_labels):
    #Map unique_list to cluster_labels
    element_cluster_map = list(zip(unique_list, cluster_labels))
    
    #Group elements by their cluster
    cluster_groups = {}
    for element, cluster in element_cluster_map:
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(element)
    
    #Create all possible pairs within each cluster
    cluster_pairs = []
    for cluster, elements in cluster_groups.items():
        if len(elements) > 1:

            pairs = list(combinations(elements, 2))
            cluster_pairs.extend(pairs)
    
    return cluster_pairs

def calculate_fraction_of_comparisons(predicted_pairs, unique_items):
    total_comparisons = len(predicted_pairs)

    N = len(unique_items)
    total_possible_comparisons = (N * (N - 1)) // 2  # Combinatorial calculation

    te_fc = total_comparisons / total_possible_comparisons if total_possible_comparisons > 0 else 0

    return te_fc

def evaluate_performance(true_pairs, predicted_pairs):
    total_actual_duplicates = len(true_pairs)  # Number of actual duplicates
    total_comparisons = len(predicted_pairs)  # Number of predicted pairs
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    normalized_true_pairs = [tuple(sorted(pair)) for pair in true_pairs]
    normalized_predicted_pairs = [tuple(sorted(pair)) for pair in predicted_pairs]

    # Now find true positives, false positives, and false negatives
    tp = [pair for pair in normalized_predicted_pairs if pair in normalized_true_pairs]  # True Positives
    fp = [pair for pair in normalized_predicted_pairs if pair not in normalized_true_pairs]  # False Positives
    fn = [pair for pair in normalized_true_pairs if pair not in normalized_predicted_pairs]  # False Negatives

    # Calculate Pair Quality
    pair_quality = len(tp) / total_comparisons if total_comparisons > 0 else 0

    # Calculate Pair Completeness
    pair_completeness = len(tp) / total_actual_duplicates if total_actual_duplicates > 0 else 0

    # Calculate F1* Measure (harmonic mean of pair quality and pair completeness)
    f1_star = 2 * (pair_quality * pair_completeness) / (pair_quality + pair_completeness) if (pair_quality + pair_completeness) > 0 else 0

    # Calculate Precision and Recall
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0

    # Calculate F1 Measure
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Output the evaluation metrics
    return {
        "pair_quality": pair_quality,
        "pair_completeness": pair_completeness,
        "f1_star": f1_star,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn)
    }

