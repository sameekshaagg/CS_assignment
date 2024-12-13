import numpy as np
import pandas as pd
import json
import time
import config
import core as c
import functions as f
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def calculate_similarity_matrix(train_unique_list, titles, features, brands, a, b):
    n = len(train_unique_list)
    similarity_matrix = np.full((n, n), 10000)  # Initialize similarity matrix with a large value (10000)

    for i in range(n):
        for j in range(i, n):  # To avoid recomputation, calculate for j >= i
            
            # Fetch the actual product indices based on train_unique_list
            product1_id = train_unique_list[i]
            product2_id = train_unique_list[j]
            
            # Map IDs to titles, features, and brands
            product1_title = titles[product1_id]  
            product2_title = titles[product2_id]
            product1_features = features[product1_id]
            product2_features = features[product2_id]
            brand1 = brands[product1_id]
            brand2 = brands[product2_id]
            
            # Calculate title similarity (use your function for title similarity)
            title_sim = f.jaccard_sim_str(product1_title, product2_title, a)
            
            # Calculate feature similarity (use your feature similarity function)
            feature_sim = f.kvp_sim(product1_features, product2_features, a, b)
            
            # Combine similarities with weights
            total_similarity = title_sim * 0.6 + feature_sim * 0.4
            
            if brand1 != brand2:
                total_similarity = 10000
                
            # Store in the matrix (since it's symmetric)
            similarity_matrix[i, j] = total_similarity
            similarity_matrix[j, i] = total_similarity  

    # Set diagonal values to 0 (distance to self should be 0)
    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix

bootstrap, hash = 5, 420
char_to_int = [ord(letter) for i, letter in enumerate('Sameeksha')]
np.random.seed(sum(char_to_int))
K, T, P, Q = range(2, 8), range(1, 11), range (2, 5), range(2, 5)
B = [0, 30, 60, 70, 84]

# Load the data from the file
with open('TVs-all-merged.json') as data:
    data_json = json.load(data)
full_data = f.unpack(data_json)
d_size = len(full_data)
preprocessed_data = c.preprocess(full_data)
titles = []
brands = []
ids = []
shops = []
urls = []
features = []
feature_values_list = []
screen_sizes = []
refresh_rates = []

# Loop through the preprocessed data
for i in range(len(preprocessed_data["model_ids_clean"])):
    titles.append(preprocessed_data["titles_clean"][i])  # Title of the product
    brands.append(preprocessed_data["brands_cleaned"][i])  # Brand of the product
    ids.append(preprocessed_data["model_ids_clean"][i])  # Model ID of the product
    shops.append(preprocessed_data["shops_clean"][i])  # Shop name
    urls.append(preprocessed_data["urls"][i])  # URL
    features.append(preprocessed_data["feature_maps_renamed"][i])
    screen_sizes.append(preprocessed_data["screens"][i])
    refresh_rates.append(preprocessed_data["refresh_rates"][i])

    
# Call the function and get the feature values list
feature_values_list = f.extract_feature_values(preprocessed_data["feature_maps_renamed"], config.target_features)
final_titles = c.combine_features_with_title(titles, feature_values_list)

results_LSH = []
for k in K:
    for b in B:
        if b != 0:
            t = (1 / b) ** (1 / (hash / b))
        else:
            t = 0
        for r in range(bootstrap):
            try:
                start_time = time.time()
                bootstrap = set([int((np.random.rand() * d_size) % d_size) for i in range(d_size)])
                bootstrap_size = len(bootstrap)
                train_indices, test_indices = list(bootstrap), [i for i in range(d_size) if i not in bootstrap]
                train_ids, test_ids = f.split_items(ids, train_indices, test_indices)
                train_titles, test_titles = f.split_items(final_titles, train_indices, test_indices)
                train_shops, test_shops = f.split_items(shops, train_indices, test_indices)
                train_brands, test_brands = f.split_items(brands, train_indices, test_indices)

                train_cps, train_aps, train_fc, train_pms = c.lsh_par(train_ids, train_titles, train_shops, k, hash, b, lsh=bool(b))

                iteration_time = round((time.time() - start_time), 1)
                results_LSH.append(
                    {'k-shingle': k, 'n_bands': b, 'threshold': t,
                     'bootstrap': r+1, 'execution time (sec)': iteration_time,
                     'fraction of comparisons': train_fc,
                     'pair quality': train_pms[0], 'pair completeness': train_pms[1], 'f1*': train_pms[2]})
                print(f'{round(train_pms[2], 4)}, time: {int(iteration_time)}s')
            finally:
                continue

results_LSH = pd.DataFrame(results_LSH)
results_LSH.to_excel('results-LSH.xlsx', index=False, float_format="%.6f")

avg_results_LSH = results_LSH.groupby(['k-shingle', 'n_bands']).mean()
max_f1_star_LSH = pd.Series.argmax(avg_results_LSH[['f1*']])
k_opt = K[max_f1_star_LSH // len(B)]  # k_opt = 6
b_opt = B[max_f1_star_LSH % len(B)]  #b_opt = 70
t_opt = (1 / b_opt) ** (1 / (hash / b_opt))

train_results_full = []
test_results_full = []
error = []
for p in P:
    for q in Q: 
        for t in T:
            print(f'F1-scores for t={t/10}:')
            for r in range(bootstrap):
                try:
                    start_time = time.time()
                    bootstrap = set([int((np.random.rand() * d_size) % d_size) for i in range(d_size)])
                    bootstrap_size = len(bootstrap)
                    train_indices, test_indices = list(bootstrap), [i for i in range(d_size) if i not in bootstrap]
                    train_ids, test_ids = f.split_items(ids, train_indices, test_indices)
                    train_titles, test_titles = f.split_items(final_titles, train_indices, test_indices)
                    train_shops, test_shops = f.split_items(shops, train_indices, test_indices)
                    train_brands, test_brands = f.split_items(brands, train_indices, test_indices)
                    train_scr_sizes, test_scr_sizes = f.split_items(screen_sizes, train_indices, test_indices)
                    train_refr_rates, test_refr_rates = f.split_items(refresh_rates, train_indices, test_indices)
                    train_features, test_features = f.split_items(features, train_indices, test_indices)
                    
                    train_cps, train_aps, train_fc, train_pms = c.lsh_par(train_ids, train_titles, train_shops, k_opt, hash, b_opt)
                    test_cps, test_aps, test_fc, test_pms = c.lsh_par(test_ids, test_titles, test_shops, k_opt, hash, b_opt)

                    train_flattened_list = [item for sublist in train_cps for item in sublist]
                    train_unique_list = list(set(train_flattened_list))
                    print(train_unique_list)
                    test_flattened_list = [item for sublist in test_cps for item in sublist]
                    test_unique_list = list(set(test_flattened_list))
                    print(test_unique_list)
                    
                    train_sim = calculate_similarity_matrix(train_unique_list, train_titles, train_features, train_brands, p, q)
                    test_sim = calculate_similarity_matrix(test_unique_list, test_titles, test_features, test_brands, p, q)
                    train_sim_x = squareform(train_sim)
                    test_sim_x = squareform(test_sim)
                    
                    threshold = t/10
                    print(threshold)
                    
                    # Perform hierarchical clustering using 'single' linkage method
                    train_Z = linkage(train_sim_x, method='single')  # No need to specify 'metric' here
                    train_clusters = fcluster(train_Z, threshold, criterion='distance')
                    
                    test_Z = linkage(test_sim_x, method='single')  # No need to specify 'metric' here
                    test_clusters = fcluster(test_Z, threshold, criterion='distance')

                    train_final_pairs = c.map_clusters_and_generate_pairs(train_unique_list, train_clusters)
                    test_final_pairs = c.map_clusters_and_generate_pairs(test_unique_list, test_clusters)

                    train_fc = c.calculate_fraction_of_comparisons(train_final_pairs, train_unique_list)
                    test_fc = c.calculate_fraction_of_comparisons(test_final_pairs, test_unique_list)
                    train_results = c.evaluate_performance(train_aps, train_final_pairs)
                    test_results = c.evaluate_performance(test_aps, test_final_pairs)
                    iteration_time = round((time.time() - start_time), 1)
                    # Append the results to the results_full list along with additional information like iteration time and comparison fraction
                    train_results_full.append({
                        'p': p, 
                        'q': q,
                        't': t,
                        'tp': train_results['tp'],
                        'fp': train_results['fp'],
                        'fn': train_results['fn'],
                        'bootstrap': r + 1,  # Bootstrap iteration (starting from 1)
                        'execution time (sec)': iteration_time,  # Time taken for this iteration
                        'fraction of comparisons': train_fc,  # Fraction of comparisons
                        'pair quality': train_results['pair_quality'],  # Pair quality from the evaluation
                        'pair completeness': train_results['pair_completeness'],  # Pair completeness from the evaluation
                        'f1*': train_results['f1_star'],  # F1* measure
                        'precision': train_results['precision'],  # Precision from the evaluation
                        'recall': train_results['recall'],  # Recall from the evaluation
                        'f1': train_results['f1']  # F1 measure
                    })
                    
                    test_results_full.append({     
                        'p': p,
                        'q': q,               
                        't': t,
                        'tp': test_results['tp'],
                        'fp': test_results['fp'],
                        'fn': test_results['fn'],
                        'bootstrap': r + 1,  # Bootstrap iteration (starting from 1)
                        'execution time (sec)': iteration_time,  # Time taken for this iteration
                        'fraction of comparisons': test_fc,  # Fraction of comparisons
                        'pair quality': test_results['pair_quality'],  # Pair quality from the evaluation
                        'pair completeness': test_results['pair_completeness'],  # Pair completeness from the evaluation
                        'f1*': test_results['f1_star'],  # F1* measure
                        'precision': test_results['precision'],  # Precision from the evaluation
                        'recall': test_results['recall'],  # Recall from the evaluation
                        'f1': test_results['f1']  # F1 measure
                    })
        
                    
                    # Optionally print the results for the current iteration
                    print(f'{round(train_results["f1"], 4)}, time: {int(iteration_time)}s')
                    print(f'{round(test_results["f1"], 4)}, time: {int(iteration_time)}s')
                except Exception as e:
                    print(f"Error occurred at iteration {r + 1}: {e}")
                    error.append({'p': p, 'q': q, 't': t, 'r': r + 1, 'error': str(e)})
                finally:
                    continue
        
train_results_full = pd.DataFrame(train_results_full)
train_results_full.to_excel('train_results-full.xlsx', index=False, float_format="%.6f")

error = pd.DataFrame(error)
error.to_excel('error-full.xlsx', index=False, float_format="%.6f")

test_results_full = pd.DataFrame(test_results_full)
test_results_full.to_excel('test_results-full.xlsx', index=False, float_format="%.6f")
