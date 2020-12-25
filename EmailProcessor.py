import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def read_file():
    spam = pd.read_csv('spam.csv', encoding='latin-1')
    spam = spam.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    spam = spam.replace(['ham', 'spam'], [0, 1])
    frame = []
    for idx, series in spam.iterrows():
        data_text = {'v1': series['v1'], 'v2': clean_up(series['v2'])}
        df = pd.DataFrame([data_text])
        frame.append(df)
    dataset = pd.concat(frame, ignore_index= True)
    print(dataset.shape)
    return dataset

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

def remove_hyperlink(text):
    return re.sub(r"http\S+", "", text)

def lower_case(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_whitespace(text):
    return text.strip()

def remove_newline(text):
    return text.replace("\n", "")

def Word_Stemmer(text):
    stemmer = PorterStemmer()
    return stemmer.stem(text)

def Word_Lemmatizer(text):
    Lemmatizer = WordNetLemmatizer()
    return Lemmatizer.lemmatize(text)

def Stop_Word_Remove(text):
    stop_words = set(stopwords.words('english'))
    word_token = word_tokenize(text)
    filter_words = [word for word in word_token if not word in stop_words]
    text = " ".join(filter_words)
    return text

def clean_up(text):
    methods = [remove_hyperlink, lower_case, remove_punctuation, remove_newline, remove_whitespace, Stop_Word_Remove, Word_Lemmatizer]
    for func in methods:
        text = func(text)
    return (text)

def process_text(text):
    vectorizer = TfidfVectorizer()
    return vectorizer.transform(text)

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

def main():
    text = clean_up("running up the 6 with the hoes, you know that shit dont matter")
    #process = process_text(text)
    read_file()
main()
