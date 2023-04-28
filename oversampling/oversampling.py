import pandas as pd
from sklearn.utils import resample

# Load the dataset into a pandas DataFrame
df = pd.read_csv('basetable.csv', sep=',')

# Separate the minority and majority classes
df_minority = df[df['CD4/8'] == 'CD8']
df_majority = df[df['CD4/8'] == 'CD4']

# Upsample the minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=123)

# Combine the majority class with the upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df1 = df_upsampled[['v_gene', 'j_gene', 'cdr3_length', 'v_deletions', 'j_deletions', 'd5_deletions',  'd3_deletions', 'n1_insertions', 'n2_insertions', 'CD4/8']].reset_index(drop=True)

df1.to_csv('./basetable_oversample.csv', sep='\t', index=False)
# Check the class distribution after upsampling
print(df_upsampled['CD4/8'].value_counts())
