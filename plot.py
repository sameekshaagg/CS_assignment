import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LSH_df = (pd.read_excel('....') #excel file from pair filtering
          .groupby(['k-shingle', 'n_bands'], as_index=False).mean())
LSH_df['n_bands'] = LSH_df['n_bands'].apply(lambda x: 420/x if x != 0 else 0)
LSH_df = LSH_df.rename(columns={'n_bands': 'rows per band'})

# Loop through the metrics: pair quality, pair completeness, and f1*
for value in ['pair quality', 'pair completeness', 'f1*']:
    plt.figure(figsize=(10, 6)) 
  
    df = LSH_df[['k-shingle', value, 'fraction of comparisons']]
  
    sns.lineplot(data=df, x='fraction of comparisons', y=value, 
                 style='k-shingle', hue='k-shingle', markers=True, dashes=False)
  
    plt.title(f'{value} vs Fraction of Comparisons', fontsize=16)
    plt.xlabel('Fraction of Comparisons', fontsize=14)
    plt.ylabel(value, fontsize=14)
  
    plt.legend(title='k-shingle')
  
    plt.savefig(f'{value}_vs_fraction_of_comparisons.jpg', format='jpeg', dpi=300)
    
    plt.show()

###########################################
# this si used for MSM plotting
df = pd.read_excel('....') #replace with excel file

LSH_df = df.groupby(['p', 'q', 't'], as_index=False).mean()

for value in ['f1', 'pair quality', 'pair completeness', and 'f1*']:
    plt.figure(figsize=(10, 6)) 
    
    df = LSH_df[['t', value, 'fraction of comparisons']]
    
    sns.lineplot(data=df, x='fraction of comparisons', y=value, 
                 hue='t', palette='viridis', markers=True, dashes=False)
    
    plt.title(f'{value} vs Fraction of Comparisons', fontsize=16)
    plt.xlabel('Fraction of Comparisons', fontsize=14)
    plt.ylabel(value, fontsize=14)
    
    plt.savefig(f'{value}hi_vs_fraction_of_comparisons.jpg', format='jpeg', dpi=300)
    
    plt.show()
