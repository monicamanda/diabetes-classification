import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

__dir__ = os.path.dirname(__file__)
print(__dir__)

def dataProcessing(fileName):
    dataset = np.loadtxt(__dir__+'/'+fileName, delimiter=',', skiprows=1)
    return dataset

def dataScaling(outputFile, dataset):
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(dataset[:, :8])
    df = pd.DataFrame({'Pregnancies': scaled_dataset[:, 0],
                       'Glucose': scaled_dataset[:, 1],
                       'BloodPressure': scaled_dataset[:, 2],
                       'SkinThickness': scaled_dataset[:, 3],
                       'Insulin': scaled_dataset[:, 4],
                       'BMI': scaled_dataset[:, 5],
                       'DiabetesPedigreeFunction': scaled_dataset[:, 6],
                       'Age': scaled_dataset[:, 7],
                       'Outcome': np.array(dataset[:, 8])
                       })
    df.to_csv(__dir__+'/diabetes-scaling.csv', index=False)

    print('.. data standardization complete ..')

def naiveBayes(dataset):
    X = dataset[:, :8]
    Y = dataset[:, 8:]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, shuffle=False)
    model_gnb = GaussianNB()
    model_gnb.fit(x_train, y_train.ravel())
    predict_gnb = model_gnb.predict(x_test)

    print('\n Naive Bayes')
    print('\n accuracy: ', round((accuracy_score(predict_gnb, y_test)*100), 2), '%')
    print('\n', classification_report(y_test, predict_gnb))
    print('===========================================================')

def decisionTree(dataset):
    X = dataset[:, :8]
    Y = dataset[:, 8:]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, shuffle=False)
    model_tree = tree.DecisionTreeClassifier()
    model_tree.fit(x_train, y_train.ravel())
    predict_tree = model_tree.predict(x_test)

    print('\n Decision Tree')
    print('\n accuracy: ', round(
        (accuracy_score(predict_tree, y_test)*100), 2), '%')
    print('\n', classification_report(y_test, predict_tree))
    print('===========================================================')

def SVM(dataset):
    X = dataset[:, :8]
    Y = dataset[:, 8:]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, shuffle=False)
    model_svm = svm.SVC(kernel='linear')
    model_svm.fit(x_train, y_train.ravel())
    predict_svm = model_svm.predict(x_test)

    print('\n Support Vendor Machine')
    print('\n accuracy: ', round((accuracy_score(predict_svm, y_test)*100), 2), '%')
    print('\n', classification_report(y_test, predict_svm))
    print('===========================================================')


if __name__ == "__main__":
    print('==========================')
    print('==         Menu         ==')
    print('==========================')
    print('1. Dataset Standardization')
    print('2. Classification Process')
    print('==========================')
    choice = int(input('nomor menu: \n'))
    if choice == 1:
        dataset = dataProcessing('diabetes.csv')
        dataScaling('diabetes-scaling.csv', dataset)
    elif choice == 2:
        dataset = dataProcessing('diabetes-scaling.csv')
        naiveBayes(dataset)
        decisionTree(dataset)
        SVM(dataset)
