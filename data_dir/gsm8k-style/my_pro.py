import pandas as pd
df=pd.read_parquet("aime2025.parquet")
print(df.iloc[0])

import pandas as pd
df=pd.read_parquet("aime2024.parquet")
print(df.iloc[0])
# for i in range(len(df)):
#     df.loc[i,'data_source']='aime2025'
#
# df.to_parquet("./aime2025.parquet")
