import pandas as pd


# #pd.read_csv('spam.csv', encoding='latin-1')
# # pd.read_csv('spam.csv', sep='|', encoding='latin-1')
# df=pd.read_csv("spam.csv", encoding='ISO-8859-1')

spam = pd.read_csv('spam.csv', encoding='latin-1')
print(spam)
spam = spam.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
print(spam)