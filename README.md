# Predictive-Analysis-of-CD4/8 ratio
Build ML classification model with selected parameters training TERI baseline\
Classification model identifies **which Cells a sequences belongs to**

## Build Classification Model Pipeline
### Load Trained Dataset and Data Overview
- A **training base table** is typically stored in a pandas dataframe. Several important variables in the basetable are : **`v_gene`**, **`j_gene`**, **`cdr3_length`**, **`v_deletions`**, **`j_deletions`** and the target CD4 or CD8 column **`CD4/8`**
- **CD4/8** is the event to **`predict`**

```python
import pandas as pd
df_cd4 = pd.read_csv('./cd4_basetable_2.csv', sep='\t')
df_cd4['CD4/8'] = 'CD4'

df_cd8 = pd.read_csv('./cd8_basetable_2.csv', sep='\t')
df_cd8['CD4/8'] = 'CD8'
df = pd.concat([df_cd4, df_cd8], ignore_index=True, sort=False)

df1 = df[['v_gene', 'j_gene', 'cdr3_length', 'v_deletions', 'j_deletions', 'CD4/8']].reset_index(drop=True)
df1.to_csv('./testtable.csv', sep=',', index=False)
```
