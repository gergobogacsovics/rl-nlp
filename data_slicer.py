# %%
import pandas as pd

df = pd.read_csv("../BloombergNRG.csv", sep=";")

# %%
training_set_length = int(df.shape[0] * 0.7)  # 70% train

df_train = df[:training_set_length]

df_train.to_csv("../BloombergNRG_train.csv", index=False)

# %%
validation_set_length = int(df.shape[0] * 0.8)  # 10% val

df_val = df[training_set_length:validation_set_length]

df_val.to_csv("../BloombergNRG_val.csv", index=False)

# %%

df_test = df[validation_set_length:]

df_test.to_csv("../BloombergNRG_test.csv", index=False)
