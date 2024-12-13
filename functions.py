import numpy as np
import pandas as pd
import re


def unpack(json_obj):
    unpacked_json = []
    for key, value in json_obj.items():
        for element in value:
            unpacked_json.append(element)

    return unpacked_json

def clean_text(text, replacements):
    text = text.lower()  # Convert to lowercase
    for old, new in replacements:
        text = text.replace(old, new)  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip() 

def split_items(data: list, indices_a: list, indices_b: list):
    output_a = [data[index] for index in indices_a]
    output_b = [data[index] for index in indices_b]
    return output_a, output_b


def extract_values(value, key):
    temp_list = []
    for i in range(len(value)):
        temp_list.append(value[i].get(key))
    return temp_list

def extract_brand_from_title(title: str, known_brands: set) -> str:
    for brand in known_brands:
        if brand.lower() in title.lower():
            return brand
    return "Unknown"

def rename_feature(feature, rename_map):
    return rename_map.get(feature, feature)

def pair_match(str_list):
    dim = len(str_list)
    matrix = np.full((dim, dim), True)
    for i in range(dim):
        for j in range(len(str_list)):
            if (str_list[i] is not None) and (str_list[j] is not None):
                matrix[i][j] = str_list[i] == str_list[j]
    return matrix

def extract_feature_values(products, target_features):
    # List to hold feature values for all products
    feature_values_list = []
    
    for product in products:
        product_values = []
    
        for feature in target_features:
            # Check if the feature exists in the product
            if feature in product:
                product_values.append(product[feature])  t
        
        # Add the list of feature values for this product to the final list
        feature_values_list.append(product_values)
    
    return feature_values_list


def shingle_matrix(str_list: list, k):
    shingle_main = set()
    for string in str_list:
        if len(string) <= k:
            shingle_main.add(string)
            break
        for i in range(len(string)+1-k):
            shingle = string[i:i+k]
            shingle_main.add(shingle)
    shingle_list = list(shingle_main)

    bool_matrix = np.full((len(shingle_main), len(str_list)), False)
    for m in range(len(shingle_list)):
        curr_shingle = shingle_list[m]
        for n in range(len(str_list)):
            desc = str_list[n]
            if len(desc) <= k:
                if curr_shingle == desc:
                    bool_matrix[m][n] = True
                continue
            if curr_shingle in desc:
                bool_matrix[m][n] = True

    return bool_matrix


def min_hash(sparse_matrix: np.ndarray, n_perm):
    n_shingles, n_items = np.shape(sparse_matrix)
    sig_matrix = np.zeros((n_perm, n_items))

    for i in range(n_perm):
        perm = np.random.permutation(sparse_matrix)
        for j in range(n_items):
            for k in range(n_shingles):
                if perm[k, j]:
                    sig_matrix[i, j] = k
                    break

    return sig_matrix


def lsh(sig_matrix, n_bands):
    sig_matrix = pd.DataFrame(sig_matrix)
    if np.shape(sig_matrix)[0] % n_bands != 0:
        print('Warning: not all rows used for bands')
    candidate = []

    for q, subset in enumerate(np.array_split(sig_matrix, n_bands, axis=0)):
        band = []
        for col in subset.columns:
            block = [str(int(signature)) for signature in subset.iloc[:, col]]
            identifier = '.'.join(block)
            band.append(identifier)

        for i in range(len(band)-1):
            for j in range(i+1, len(band)):
                if band[i] == band[j]:
                    candidate.append((i, j))

    candidate_list = list(set(candidate))
    return candidate_list

def jaccard_sim_str(str1, str2, k):
    strings_shingled = shingle_matrix([str1, str2], k)
    bool_vector1, bool_vector2 = strings_shingled[:, 0], strings_shingled[:, 1]

    intersection = sum([int(bin_value) for i, bin_value in enumerate(bool_vector1) if bool_vector2[i]])
    union = len(bool_vector1)
    return intersection/union


def feature_sim(f_set1: dict, f_set2: dict, k, q):
    key_set1, key_set2 = f_set1.keys(), f_set2.keys()
    key_similarity = np.array([jaccard_sim_str(key1, key2, k) for key1 in key_set1 for key2 in key_set2])
    val_similarity = np.array([jaccard_sim_str(f_set1[key1], f_set2[key2], q) for key1 in key_set1 for key2 in key_set2])
    similarity, total_weight = sum(key_similarity*val_similarity), sum(key_similarity)

    if total_weight == 0:
        return 0

    return similarity / total_weight
