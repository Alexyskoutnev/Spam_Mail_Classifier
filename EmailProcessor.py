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
    df_main = pd.DataFrame(columns=['crl.tot', 'dollar', 'bang', 'money', 'n000', 'make', 'free', "click", "now", "cash", "quick", "easy", "offer", "order", "email", "time", "credit", "address", "people", 'spam'])
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
    click_count = x.count("click")
    now_count = x.count("now")
    cash_count = x.count("cash")
    quick_count = x.count("quick")
    easy_count= x.count("easy")
    offer_count = x.count("offer")
    order_count = x.count("order")
    email_count = x.count("email")
    time_count = x.count("time")
    credit_count = x.count("credit")
    address_count = x.count("address")
    people_count = x.count("people")
    return {"crl.tot": capital_num, 'dollar': dollar_count, 'bang': bang_count, 'money': money_count, 'n000': zero_count, 'make': make_count, 'free': free_count, 'click': click_count, 'now': now_count, 'cash': cash_count, 'quick': quick_count, 'easy': easy_count, 'offer': offer_count, 'order':order_count, 'email': email_count, 'time': time_count, 'credit': credit_count, 'address': address_count, 'people': people_count}