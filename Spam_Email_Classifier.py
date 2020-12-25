import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from EmailProcessor import read_train
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from EmailProcessor import read_file
from sklearn.naive_bayes import GaussianNB
from EmailProcessor import clean_up

#spam = pd.read_csv('spambase.data')

#Split Data into train and test subsets
def train_data(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, y_train

def test_data(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_test, y_test

def process_data():
    Dataset = read_file()
    y_train, x_train = train_data(Dataset)
    y_test, x_test = test_data(Dataset)
    x_train = [txt.split(" ") for txt in x_train]
    x_test = [txt.split(" ") for txt in x_test]
    x_train_features = Vectorize_data(x_train)
    x_test_features = Vectorize_data(x_test)
    return x_train_features.toarray(), x_test_features.toarray(), y_train, y_test


def Vectorize_data(data):
    vectorizer = TfidfVectorizer()
    sentences = [' '.join(txt) for txt in data]
    vectorizer.fit(sentences)
    return vectorizer.transform(sentences)

def data_features(data):
    vectorizer = TfidfVectorizer()
    raw_sentences = [' '.join(o) for o in data]
    return vectorizer.transform(raw_sentences)

# def process_data(data):
#     #Message = X.split(" ")
#     vectorizer = TfidfVectorizer()
#     return vectorizer.transform(data)

#Gaussian Regression
def Gaussian_Regression(X,Y):
    model = GaussianNB()
    model.fit(X,Y)
    return model

#LogisticRegression
def Logistic_Regression(X,Y, dataset):
        C_range = [0.01, .1, 1, 10, 100, 1000]
        test_accuracy = -1
        for i in C_range:
            model = LogisticRegression(C = i, random_state=0, max_iter= 1000000).fit(X, Y)
            test_result = model.score(*test_data(dataset))
            if (test_result > test_accuracy):
                test_accuracy = test_result
                C_max = i
        model = LogisticRegression(C = C_max, random_state=0).fit(X, Y)
        return model

def Logistic_Model(Dataset):
    Log_Model = Logistic_Regression(*train_data(Dataset), Dataset)
    return Log_Model

#Support Vector Machine
def Support_Vector_Machine(X,Y, dataset):
    kernel_type = ['rbf']
    C_range = [0.01, 1, 10, 100]
    test_accuracy = -1
    for type in kernel_type:
        for C_type in C_range:
            model = svm.SVC(kernel= type, C = C_type, max_iter= 1000000)
            model.fit(X, Y)
            test_result = model.score(*test_data(dataset))
            if (test_result > test_accuracy):
                test_accuracy = test_result
                C_max = C_type
                kernel_max = type
    model = svm.SVC(C = C_max, kernel = kernel_max)
    model.fit(X, Y)
    return model

def Support_Vector_Model(Dataset):
    SVM_Model = Support_Vector_Machine(*train_data(Dataset), Dataset)
    return SVM_Model

#Decision Tree
def Decision_Tree(X, Y):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X,Y)
    return model

def Decision_Tree_Model(Dataset):
    Decision_Model = Decision_Tree(*train_data(Dataset))
    return Decision_Model

def result(Dataset):
    Decision_Model = Decision_Tree(*train_data(Dataset))
    print("Decision Tree Training set score: {:.3f}".format(Decision_Model.score(*train_data(Dataset))))
    print("Decision Tree Test set score: {:.3f}".format(Decision_Model.score(*test_data(Dataset))))
    Log_Model = Logistic_Regression(*train_data(Dataset), Dataset)
    print("Logistic Training set score: {:.3f}".format(Log_Model.score(*train_data(Dataset))))
    print("Logistic Test set score: {:.3f}".format(Log_Model.score(*test_data(Dataset))))
    SVM_Model = Support_Vector_Machine(*train_data(Dataset), Dataset)
    print("SVM Training set score: {:.3f}".format(SVM_Model.score(*train_data(Dataset))))
    print("SVM Test set score: {:.3f}".format(SVM_Model.score(*test_data(Dataset))))
def Spam_printer(w, type, name):
    """
    w writes to the console if
    text is spam or not
    """
    if (type[0] == 0):
        w.write(str(name) + ": Not Spam" + "\n")
    elif (type[0] == 1):
        w.write(str(name)+ ": Spam" + "\n")

def spam_solve(r, w):
    """
    r a reader
    w a writer
    """
    Dataset = read_train()
    lines = r.readlines()
    MSG = ''
    for x in lines:
        MSG += x.replace("\n", " ")
    test_output = read_file(MSG)
    Log_Model = Logistic_Model(Dataset)
    Spam_printer(w, Log_Model.predict(test_output), LogisticRegression.__name__)
    Decision_Model = Decision_Tree_Model(Dataset)
    Spam_printer(w, Decision_Model.predict(test_output), RandomForestClassifier.__name__)
    SVM_Model = Support_Vector_Model(Dataset)
    Spam_printer(w, SVM_Model.predict(test_output), svm.SVC.__name__)
    #result(Dataset)

def solver(r,w):
    Dataset = read_file()
    lines = r.readlines()
    MSG = ''
    for x in lines:
        MSG += x.replace("\n", " ")
    test_output = clean_up(MSG).split(" ")
    test_output_features = Vectorize_data(test_output)
    x_train, x_test, y_train, y_test = process_data()
    Gaussian_Model = Gaussian_Regression()
    pass

def main():
    Dataset = read_file()
    y_train, x_train = train_data(Dataset)
    y_test, x_test = test_data(Dataset)
    x_train = [o.split(" ") for o in x_train]
    print(y_train[0])
    print(x_train[0])
    print(len(x_test))
    print(len(x_train))
    x_train_features = Vectorize_data(x_train)
    print(x_train_features)
    x_train, x_test, y_train, y_test = process_data()
    print(len(x_train[0]))
    print(y_train[0])
    Gaussian_Model = Gaussian_Regression(x_train, y_train)
    print("Gaussian Model Training set score: {:.3f}".format(Gaussian_Model.score(x_train, y_train)))
    print("Guassian Model Test set score: {:.3f}".format(Gaussian_Model.score(x_test, y_test)))

    pass

main()