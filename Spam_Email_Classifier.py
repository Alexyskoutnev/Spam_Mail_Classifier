import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

spam = pd.read_csv('spambase.data')

#Split Data into train and test subsets
def train_data(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    return (x_train, y_train)


#LogisticRegression
def Logistic_Regression(X,Y):
        model = LogisticRegression(solver='liblinear', random_state=0).fit(X, Y)
        return model

        # print(model.intercept_)
        # print(model.coef_)
def main():
    pass
    # print("Logistic Training set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))
    # print("Logistic Test set score: {:.3f}".format(LogisticRegression.score(*train_data(spam))))

Logistic_Regression(*train_data(spam))
main()
print(*train_data(spam))
