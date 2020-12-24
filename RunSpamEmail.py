import pandas as pd
import sys
from Spam_Email_Classifier import train_data
from Spam_Email_Classifier import test_data
from EmailProcessor import read_train
from Spam_Email_Classifier import result
from Spam_Email_Classifier import Decision_Tree
from Spam_Email_Classifier import Support_Vector_Machine
from Spam_Email_Classifier import LogisticRegression
from Spam_Email_Classifier import Decision_Tree_Model
from Spam_Email_Classifier import Support_Vector_Model
from Spam_Email_Classifier import Logistic_Model
from Spam_Email_Classifier import spam_solve


def Run_Classifier():
    Dataset = read_train()
    # result(Dataset)
    Log_Model = Logistic_Model(Dataset)
    #SVM_Model = Support_Vector_Model(Dataset)
    # Decision_Model = Decision_Tree_Model(Dataset)
    #print(SVM_Model.predict([[1, 0, 0, 0, 0, 0, 0]]))

if __name__ == "__main__":
    spam_solve(sys.stdin, sys.stdout)
#Run_Classifier()