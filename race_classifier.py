import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

#cleaned data contains: name, gender and race
df = pd.read_csv("final_names.csv")
df.drop(["Unnamed: 0", "gender"], axis=1, inplace=True)

#for multiclass convert all the races into vector format
race_list = []
n = len(df)
race = df["race"]
for i in range(n):
    if race[i] == "black":
        race_list.append(np.array((1,0,0,0)))
    elif race[i] == "white":
        race_list.append(np.array((0,1,0,0)))
    elif race[i] == "hispanic":
        race_list.append(np.array((0,0,1,0)))
    elif race[i] == "indian":
        race_list.append(np.array((0,0,0,1)))
df["race"] = race_list

def myfeatures(name):
    name = name.lower()
    return {
        "first_letter": name[0],
        "first_two_letters": name[0:2],
        "first_three_letters": name[0:3],
        "last_three_letters": name[-3:],
        "last_two_letters": name[-2:],
        "last_letter": name[-1],
    }
myfeatures = np.vectorize(myfeatures)

df_matrix = df.as_matrix()
X = myfeatures(df_matrix[:,0].astype(np.str))
Y = df_matrix[:,1].astype(list)
Y = np.vstack(Y)
X, Y = shuffle(X, Y)

X_train = X[:int(len(X)*.8)]
Y_train = Y[:int(len(X)*.8)]
X_test = X[int(len(X)*.8):]
Y_test = Y[int(len(X)*.8):]

## Decision Tree Classifier
vect = DictVectorizer()
dtc = DecisionTreeClassifier()
model = Pipeline([('dict', vect), ('dtc', dtc)])
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
accuracy = np.mean((prediction == Y_test))
accuracy = round(accuracy, 4)
print("Accuracy of the Dicision Tree Classifier (on Test data set):\n", 100*(accuracy), "%") 

#getting precision and recall on test data set
P, R, F1_score, _ =  prfs(Y_test, prediction, average="micro")

print("Pricision, Recall and F score on test set:\nPricision:", P, "\nRecal:", R, "\nF1 Score:", F1_score)

#Model
#return race for a given name array
def getRace(name):
    name = myfeatures(name)
    pred_arr = model.predict(name).ravel()
    return(pred_arr)

print("cheers")