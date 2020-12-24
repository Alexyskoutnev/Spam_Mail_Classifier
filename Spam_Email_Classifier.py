import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from EmailProcessor import read_file
from EmailProcessor import read_train
from sklearn.metrics import classification_report, confusion_matrix

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

#LogisticRegression
def Logistic_Regression(X,Y, dataset):
        C_range = [0.0001, 0.001, 0.01, .1, 1, 10, 100, 1000, 10000]
        test_accuracy = -1
        for i in C_range:
            model = LogisticRegression(C = i, random_state=0, max_iter= 10000000).fit(X, Y)
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
    model = RandomForestClassifier(n_estimators=50, random_state=0)
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
def Spam_printer(w, type):
    if (type[0] == 0):
        w.write("Not Spam")
        print()
    elif (type[0] == 1):
        w.write("Spam")
        print()

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
    x_test, y_test = train_data(Dataset)
    # print(test_output)
    # print(x_test[1])
    # print(y_test[1])
    #print(Logistic_Regression(*train_data(Dataset), Dataset).predict(test_output))
    Log_Model = Logistic_Model(Dataset)
    Spam_printer(w, Log_Model.predict(test_output))
    Decision_Model = Decision_Tree_Model(Dataset)
    Spam_printer(w, Decision_Model.predict(test_output))
    SVM_Model = Support_Vector_Model(Dataset)
    Spam_printer(w, SVM_Model.predict(test_output))

def main():
    pass
    # print("SVM Training set score: {:.3f}".format(Support_Vector_Machine.score(*train_data(spam))))
    # print("SVM Test set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))
    #print(LogisticRegression(*train_data(spam)).score(*test_data(spam)))
    # print("Logistic Training set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))
    # print("Logistic Test set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))
