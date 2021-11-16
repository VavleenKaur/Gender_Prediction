
# Vavleen Kaur 
# LOGISTIC REGRESSION
#GENDER PREDICTION
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Unlike linear regression, where we want to predict a continuous value, we want our classifier to predict the probability that the data is positive (1), or negative (0). For this we will use the Sigmoid function:
# g(z) = {1 \ 1 + e^{-z}}


def plotSigmoid():
    x = np.arange(-50, 50)
    y = np.exp(x)/(1+np.exp(x))
    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# If we plot the function, we will notice that as the input approaches infinity, the output approaches 1, and as the input approaches infinity, the output approaches 0.


def importDataset():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/johnmyleswhite/ML_for_Hackers/master/02-Exploration/data/01_heights_weights_genders.csv")
    print(df.info())
    x = df.iloc[:, 1:3]
    y = df.iloc[:, 0]
    return x, y, df


def heatmap(df):
    sns.heatmap(df.iloc[:, :5].corr(), annot=True)
    plt.show()


def MissingNo(df):
    msn.bar(df)
    plt.show()


def trainTestSplit(x, y):
    return train_test_split(x, y, random_state=42, test_size=0.3)


# One more consideration we have to make before writing our training function is that our current classification method only works with two class labels: positive and negative. In order to classify more than two labels, we will employ whats known as one-vs.-rest strategy: For each class label we will fit a set of parameters where that class label is positive and the rest are negative.
def trainModel(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model


def testModel(model, x_test):
    return model.predict(x_test)


def cnMatrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    print(f'Accuracy is : ', accuracy)
    print(f'Precision is : ', precision)
    print(f'Recall is : ', recall)


def main():
    plotSigmoid()
    x, y, df = importDataset()
    heatmap(df)
    MissingNo(df)
    x_train, x_test, y_train, y_test = trainTestSplit(x, y)
    #plt.scatter(x.iloc[:, 1], x.iloc[:, 2], c=y.iloc[:,1], alpha=0.5)
    model = trainModel(x_train, y_train)
    y_pred = testModel(model, x_test)
    cMatrix(y_test, y_pred)


if __name__ == "__main__":
    main()
