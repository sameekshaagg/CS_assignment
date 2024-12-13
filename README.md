# CS_assignment
Duplicate Detection Assignment (LSH and MSM)

The config file is used in standardizing product data, improving consistency, and preparing for further processing tasks such as clustering or feature extraction.
1. Brand and Shop Sets: Lists of recognized brands and shops used for data processing.
2. Replacements: A list of characters or terms (like hyphens, slashes, or "Yes") to be replaced during text preprocessing for consistency.
3. Feature Rename Map: A dictionary mapping feature names in the dataset to standardized names (e.g., 'energy consumption' to 'power consumption').
4. Target Features: A list of features to focus on, such as "energy star certified" and "screen size".

The main file has multiple functions: 
1. Preprocessing: The dataset is first preprocessed by cleaning and extracting relevant features such as titles, brands, and product features.
2. Similarity Matrix Calculation: A similarity matrix is created based on the Jaccard similarity of product titles and feature matching.
3. Bootstrapping and Evaluation: The dataset is divided into bootstrap samples for training and testing. Various configurations of LSH parameters (shingle size, number of bands) are evaluated using F1-score, precision, recall, and other metrics.
4. Clustering: Hierarchical clustering is applied to the similarity matrix to group similar products.
5. Results Export: The results, including performance metrics and clustering information, are stored in Excel files for further analysis.

The core_functions consists of the main function called in the main file
1. Preprocessing: Cleans and extracts product details like modelID, shop, featuresMap, title, and url. It also handles text cleaning, brand extraction, and feature renaming.
2. LSH Pair Filtering: Uses Locality Sensitive Hashing (LSH) to identify candidate product pairs, compares them to actual duplicates, and calculates pair quality and comparison fractions.
3. Feature Combination: Adds feature values to product titles, ensuring no duplication of words.
4. Cluster Pair Generation: Groups products by cluster labels and creates all possible pairs within each cluster.
5. Comparison Fraction: Calculates the fraction of comparisons relative to the total possible pairs.
6. Performance Evaluation: Measures model performance using metrics like precision, recall, F1 score, and pair quality, comparing predicted pairs to true duplicates.

The function fill contains all the supporting functions needed for the core_functions, like calculating LSH, min-hash, shingling, rename, pairing etc. 

The plot creates the plot with a fraction of comparison and different values, like F1, F1*, pair quality and pair completeness.
