import pandas as pd

spam = pd.read_csv('spambase.data')
print(spam.shape)
print(spam.columns)