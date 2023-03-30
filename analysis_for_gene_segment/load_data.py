import pandas as pd

# load traing data
df_cd4 = pd.read_csv('E:/Document/TERI_baseline_normalized/4012V1_CD4.tsv', sep='\t')
df_cd4['CD4/8'] = 'CD4'

df_cd8 = pd.read_csv('E:/Document/TERI_baseline_normalized/4012V1_CD8.tsv', sep='\t')
df_cd8['CD4/8'] = 'CD8'

df = pd.concat([df_cd4, df_cd8], ignore_index=True, sort=False)
# print(df.columns.values)

df = df[df['v_gene'].str.contains('N') == False]
df = df[df['j_gene'].str.contains('N') == False]
df = df[df['v_gene'].str.contains('OR') == False]
df1 = df[['amino_acid', 'CD4/8']].reset_index(drop=True)  # 'd5_deletions', 'd3_deletions',
print(df1)

sum = len(df_cd4) + len(df_cd8)
print(sum)
# p_exact = len(df_cd4) / len(df)
# p_exact_1 = len(df_cd4) / len(df1)
p_exact_2 = len(df_cd4) / sum
# print('****', p_exact)
# print('####', p_exact_1)
print('++++', p_exact_2)
# df1.to_csv('E:/Document/28_03_v_gene/train_amino_acid/4012V1_amino_acid.csv', sep='\t', index=False)
