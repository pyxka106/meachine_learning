import pandas as pd

# testing data
df_cd4 = pd.read_csv('E:/Document/healthy_subject_normalized/Healthy_Subject_7_CD4_Naive.tsv', sep='\t')
df_cd4['CD4/8'] = 'CD4'

df_cd8 = pd.read_csv('E:/Document/healthy_subject_normalized/Healthy_Subject_7_CD8_Naive.tsv', sep='\t')
df_cd8['CD4/8'] = 'CD8'

df = pd.concat([df_cd4, df_cd8], ignore_index=True, sort=False)

df = df[df['v_gene'].str.contains('N') == False]
df = df[df['j_gene'].str.contains('N') == False]
df = df[df['v_gene'].str.contains('OR') == False]
df1 = df[['v_gene', 'j_gene', 'cdr3_length', 'v_deletions', 'j_deletions', 'd5_deletions', 'd3_deletions', 'n1_insertions', 'n2_insertions', 'CD4/8']].reset_index(drop=True)
# print(df1)

sum = len(df_cd4) + len(df_cd8)
print(sum)
p_exact = len(df_cd4) / len(df)
# p_exact_2 = len(df_cd4) / sum
print('++++', p_exact)
df1.to_csv('./test/HD_7N.csv', sep=',', index=False)
