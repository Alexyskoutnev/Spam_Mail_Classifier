import pandas as pd
from collections import Counter

def read_file(MSG):
    data = pd.DataFrame([email_decoder(MSG)])
    X = data.values
    return X

def read_train():
    spam = pd.read_csv('spam.csv', encoding='latin-1')
    spam = spam.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    spam = spam.replace(['ham', 'spam'], [0, 1])
    df_main = pd.DataFrame(columns=['crl.tot', 'dollar', 'bang', 'money', 'n000', 'make', 'free', 'spam'])
    frame = []
    for idx, series in spam.iterrows():
        data_dict = email_decoder(series['v2'])
        data_dict['spam'] = series['v1']
        df = pd.DataFrame([data_dict])
        frame.append(df)
    dataset = pd.concat(frame, ignore_index= True)
    return dataset

def email_decoder(data):
    capital_num = sum(1 for c in data if c.isupper())
    dollar_count = data.count("$")
    bang_count = data.count("!")
    x = data.lower()
    money_count = x.count("money")
    zero_count = x.count("000")
    make_count = x.count("make")
    free_count = x.count("free")
    return {"crl.tot": capital_num, 'dollar': dollar_count, 'bang': bang_count, 'money': money_count, 'n000': zero_count, 'make': make_count, 'free': free_count}