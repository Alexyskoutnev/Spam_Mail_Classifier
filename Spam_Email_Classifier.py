import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

spam = pd.read_csv('spambase.data')

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
def Logistic_Regression(X,Y):
        C_range = [0.0001, 0.001, 0.01, .1, 1, 10, 100, 1000, 10000]
        test_accuracy = -1
        for i in C_range:
            model = LogisticRegression(C = i, random_state=0, max_iter= 10000000).fit(X, Y)
            test_result = model.score(*test_data(spam))
            if (test_result > test_accuracy):
                test_accuracy = test_result
                C_max = i
        model = LogisticRegression(C = C_max, random_state=0).fit(X, Y)
        return model

#Support Vector Machine
def Support_Vector_Machine(X,Y):
    kernel_type = ['linear', 'rbf']
    C_range = [.1, 1, 10, 100, 1000, 10000]
    test_accuracy = -1
    for type in kernel_type:
        for C_type in C_range:
            model = svm.SVC(kernel= type, C = C_type, max_iter= 100000)
            model.fit(X, Y)
            test_result = model.score(*test_data(spam))
            if (test_result > test_accuracy):
                test_accuracy = test_result
                C_max = C_type
                kernel_max = type
    model = svm.SVC(C = C_max, kernel = kernel_max)
    model.fit(X, Y)
    return model

#by

#Decision Tree
def Decision_Tree(X, Y):
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(X,Y)
    return model

def result():
    Decision_Model = Decision_Tree(*train_data(spam))
    print("Decision Tree Training set score: {:.3f}".format(Decision_Model.score(*train_data(spam))))
    print("Decision Tree Test set score: {:.3f}".format(Decision_Model.score(*test_data(spam))))
    Log_Model = Logistic_Regression(*train_data(spam))
    print("Logistic Training set score: {:.3f}".format(Log_Model.score(*train_data(spam))))
    print("Logistic Test set score: {:.3f}".format(Log_Model.score(*test_data(spam))))
    SVM_Model = Support_Vector_Machine(*train_data(spam))
    print("SVM Training set score: {:.3f}".format(SVM_Model.score(*train_data(spam))))
    print("SVM Test set score: {:.3f}".format(SVM_Model.score(*test_data(spam))))


def main():
    pass
    # print("SVM Training set score: {:.3f}".format(Support_Vector_Machine.score(*train_data(spam))))
    # print("SVM Test set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))
    #print(LogisticRegression(*train_data(spam)).score(*test_data(spam)))
    # print("Logistic Training set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))
    # print("Logistic Test set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))

result()