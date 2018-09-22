import os
import pandas as pd
import nltk
from nltk import word_tokenize
import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

#Complete Data set contains first, last name, gender and race
blackf = pd.read_csv("Black-Female-Names.csv")
blackm = pd.read_csv("Black-Male-Names.csv")
whitef = pd.read_csv("White-Female-Names.csv")
whitem = pd.read_csv("White-Male-Names.csv")
hispf = pd.read_csv("Hispanic-Female-Names.csv")
hispm = pd.read_csv("Hispanic-Male-Names.csv")
indianf = pd.read_csv("Indian-Female-Names.csv")
indianm = pd.read_csv("Indian-Male-Names.csv")

indianf["first name"] = indianf["name"]
indianm["first name"] = indianm["name"]

#Concatenate into one Data-Frame
frames = [blackf, blackm.iloc[:round(len(blackm)/4)], whitef, whitem[:round(len(whitem)/3)], hispf, hispm, indianf, indianm]
df = pd.concat(frames, ignore_index=True,verify_integrity=False)
df["first name"].fillna(value= df[" first name"], inplace=True)
df.drop([" first name", "last name", "name"], axis=1, inplace=True)
df.dropna(inplace=True)

def processWord(words_list):
    final_words_list = []
    for word in words_list:
        word = str(word)
        if len(word) < 1:
            final_words_list.append(word)
        else:
            word = word.lower()
            word = re.sub('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', " ", word)
            word = re.sub(r'[^a-zA-Z0-9]', ' ', word)
            word = re.sub( '\s+', ' ', word).strip()
            final_words_list.append(word)
    return final_words_list

#Convert the names into sensible format/words cleaning
names = df["first name"].tolist()
first_names = []
names = processWord(names)
for name in names:
    name = name.split(" ")
    if len(name)<1:
        first_names.append(np.nan)
    elif len(name) == 1:
        first_names.append(name[0])
    else:
        if len(name[0]) <= 3:
            first_names.append("".join(name[0:-1]))
        else:
            first_names.append(name[0])
final_names = []
for name in first_names:
    if len(name) <= 2:
        final_names.append(np.nan)
    else:
        final_names.append(name)

df["first name"] = final_names
df.reset_index(inplace=True, drop=True)
df.dropna(inplace=True, how="any")
df.isnull().sum()

#replace the class type of gender column; gender : male = 0 and female = 1
df["gender"] = (df["gender"]=="f").astype(int)

#NOTE: Labeled data for decision tree classifier to prevent overfitting(can be done in more better ways)
df2 = df[df["gender"]==1]
df = pd.concat([df2,df, df2,], ignore_index=True,verify_integrity=False)
df.reset_index(inplace=True, drop=True)

def myfeatures(x):
    x = x.lower()
    return {
        "x1": x[0],
        "x2": x[0:2],
        "x3": x[0:3],
        "x4": x[-3:],
        "x5": x[-2:],
        "x6": x[-1],
    }
myfeatures = np.vectorize(myfeatures)

df_matrix = df.as_matrix()
X = myfeatures(df_matrix[:,0])
Y = df_matrix[:,1].astype(np.int)
X, Y = shuffle(X, Y)

X_train = X[:round(len(X)*.6)]
Y_train = Y[:round(len(X)*.6)]
X_cv = X[round(len(X)*.6):round(len(X)*.8)]
Y_cv = Y[round(len(X)*.6):round(len(X)*.8)]
X_test = X[round(len(X)*.8):]
Y_test = Y[round(len(X)*.8):]


##############################    1. Decision Tree Classifier     ############################

vect = DictVectorizer()
dtc = DecisionTreeClassifier()
model = Pipeline([('dict', vect), ('dtc', dtc)])
model.fit(X_train, Y_train)
prediction = model.predict(X_cv)
accuracy = np.mean((prediction == Y_cv.ravel()))
accuracy = round(accuracy, 4)
print("Accuracy of the Dicision Tree Classifier (on CV data set):\n", 100*(accuracy), "%") 

#confidence interval 
classification_error = np.mean((prediction != Y_cv.ravel()))
error = classification_error
n = len(X_test)
c_1 = error - 1.96*np.sqrt( (error * (1 - error)) / n)
c_2 = error + 1.96*np.sqrt( (error * (1 - error)) / n)
confidence_interval = [c_1, c_2]

def fScore(x,y,pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(x)):
        if (y[i] == 1 & pred[i] == 1):
            TP = TP+1
        elif (y[i] == 1 & pred[i] == 0):
            FN = FN+1
        elif (y[i] == 0 & pred[i] == 0):
            TN = TN+1
        else:
            FP = FP+1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1_score = 2*P*R/(P+R)
    return(P, R, F1_score)

pred = model.predict(X_cv)
P, R, F1_score = fScore(X_cv, Y_cv, pred)
print("Precision: " , round(P, 3) , "\nRecall: ", round(R, 3), "\nF1_Score: ", round(F1_score, 3))

#Model (Giving a name it returns the gender for that name)
def getGender(name):
    name = myfeatures(name)
    return(model.predict(name))

# getGender(["Amit", "Priyaka", "ankita", "adam", "Joy", "robert"])

# array([0, 1, 1, 0, 1, 0])

########################### 2. Naive Bayes Classifier !! Trained on Complete Data     ##########################################

from nltk import NaiveBayesClassifier as nbc
from sklearn.naive_bayes import MultinomialNB
from nltk import classify
from nltk.metrics.scores import precision, recall
import collections
import nltk.metrics

df1 = pd.read_csv("final_names.csv")
df1.drop(["Unnamed: 0"], axis=1, inplace=True)
df1 = df1.as_matrix()
X1 = myfeatures(df1[:,0].astype(np.str))
Y1 = df1[:,1].astype(np.int)
X1, Y1 = shuffle(X1, Y1)

featureset = [(X1[i],Y1[i]) for i in range(len(X1))]
train_set = featureset[: int(0.7*len(featureset))]
test_set = featureset[int(0.7*len(featureset)):]

nb_clf = nbc.train(train_set)
accu = classify.accuracy(nb_clf, test_set)
accu= round(accu, 3)
print("Accuracy of the Naive Bayes Classifier (on test data set):\n", 100*(accu), "%") 

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = nb_clf.classify(feats)
    testsets[observed].add(i)

nb_P = precision(refsets[0], testsets[0])
nb_R = recall(refsets[0], testsets[0])
nb_F1_score = 2*nb_P*nb_R/(nb_P+nb_R)
print("Precision: " , round(nb_P, 2) , "\nRecall: ", round(nb_R, 2), "\nF1_Score for Naive Bayes Classifier: ", round(nb_F1_score, 2))

print("cheers")

