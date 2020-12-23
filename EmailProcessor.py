import pandas as pd
from collections import Counter

def read_train():
    spam = pd.read_csv('spam.csv', encoding='latin-1')
    spam = spam.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    spam = spam.replace(['ham', 'spam'], [0, 1])
    df = pd.DataFrame(columns=['crl.tot', 'dollar', 'bang', 'money', 'n000', 'make', 'free', 'spam'])
    for x in spam.v2:
        dataframe = email_decoder(x)


def email_decoder(data, df):
    capital_num = sum(1 for c in data if c.isupper())
    dollar_count = data.count("$")
    bang_count = data.count("!")
    x = data.lower()
    money_count = x.count("money")
    zero_count = x.count("000")
    make_count = x.count("make")
    free_count = x.count("free")
    pass