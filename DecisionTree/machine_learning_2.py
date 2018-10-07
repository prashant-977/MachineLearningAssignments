import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from tkinter import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
import sys
import getopt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=';', encoding='utf-8')

df.apply(pd.to_numeric, errors='ignore')

# Replace boolean-type string with integers.
df.gender.replace(['Male', 'Female'], [1, 0], inplace=True)
df.Partner.replace(['Yes', 'No'], [1, 0], inplace=True)
df.Dependents.replace(['Yes', 'No'], [1, 0], inplace=True)
df.PhoneService.replace(['Yes', 'No'], [1, 0], inplace=True)
df.PaperlessBilling.replace(['Yes', 'No'], [1, 0], inplace=True)

# categoricalColumns = [
#     "StreamingTV",
#     "StreamingMovies",
#     "OnlineBackup",
#     "OnlineSecurity",
#     "DeviceProtection",
#     "TechSupport",
#     "MultipleLines",
#     "Contract",
#     "InternetService",
#     "PaymentMethod"
# ]
# Convert columns to categorical values and replace value with one hot encoded one so it is more realistic.
# def oneHotEncode(df, cols):
#     for i in cols:
#         df.i = pd.Categorical(df.i)
#         df[i] = df.i.cat.codes
#         encoded = pd.get_dummies(df[i], prefix=i)
#         df = df.drop(i, axis=1)
#         df = df.join(encoded)
#     return df

# df = oneHotEncode(df, categoricalColumns)

df.StreamingTV = pd.Categorical(df.StreamingTV)
df['StreamingTV'] = df.StreamingTV.cat.codes
encoded = pd.get_dummies(df['StreamingTV'], prefix='StreamingTV')
df = df.drop('StreamingTV', axis=1)
df = df.join(encoded)

df.StreamingMovies = pd.Categorical(df.StreamingMovies)
df['StreamingMovies'] = df.StreamingMovies.cat.codes
encoded = pd.get_dummies(df['StreamingMovies'], prefix='StreamingMovies')
df = df.drop('StreamingMovies', axis=1)
df = df.join(encoded)

df.OnlineBackup = pd.Categorical(df.OnlineBackup)
df['OnlineBackup'] = df.OnlineBackup.cat.codes
encoded = pd.get_dummies(df['OnlineBackup'], prefix='OnlineBackup')
df = df.drop('OnlineBackup', axis=1)
df = df.join(encoded)

df.OnlineSecurity = pd.Categorical(df.OnlineSecurity)
df['OnlineSecurity'] = df.OnlineSecurity.cat.codes
encoded = pd.get_dummies(df['OnlineSecurity'], prefix='OnlineSecurity')
df = df.drop('OnlineSecurity', axis=1)
df = df.join(encoded)

df.DeviceProtection = pd.Categorical(df.DeviceProtection)
df['DeviceProtection'] = df.DeviceProtection.cat.codes
encoded = pd.get_dummies(df['DeviceProtection'], prefix='DeviceProtection')
df = df.drop('DeviceProtection', axis=1)
df = df.join(encoded)

df.TechSupport = pd.Categorical(df.TechSupport)
df['TechSupport'] = df.TechSupport.cat.codes
encoded = pd.get_dummies(df['TechSupport'], prefix='TechSupport')
df = df.drop('TechSupport', axis=1)
df = df.join(encoded)

df.MultipleLines = pd.Categorical(df.MultipleLines)
df['MultipleLines'] = df.MultipleLines.cat.codes
encoded = pd.get_dummies(df['MultipleLines'], prefix='MultipleLines')
df = df.drop('MultipleLines', axis=1)
df = df.join(encoded)

df.Contract = pd.Categorical(df.Contract)
df['Contract'] = df.Contract.cat.codes
encoded2 = pd.get_dummies(df['Contract'], prefix='Contract')
df = df.drop('Contract', axis=1)
df = df.join(encoded2)

df.InternetService = pd.Categorical(df.InternetService)
df['InternetService'] = df.InternetService.cat.codes
encoded3 = pd.get_dummies(df['InternetService'], prefix='InternetService')
df = df.drop('InternetService', axis=1)
df = df.join(encoded3)

df.PaymentMethod = pd.Categorical(df.PaymentMethod)
df['PaymentMethod'] = df.PaymentMethod.cat.codes
encoded4 = pd.get_dummies(df['PaymentMethod'], prefix='PaymentMethod')
df = df.drop('PaymentMethod', axis=1)
df = df.join(encoded4)

# move Churn to first Index.
cols = list(df)
cols.insert(0, cols.pop(cols.index('Churn')))
df = df.ix[:, cols]

X = df.values[:, 2:41] #Skipping second column (customerID@index=1) since data is completely irrelevant.
Y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                  max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                     max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

y_pred_en = clf_entropy.predict(X_test)

dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("gini_tree.png")

print("Gini index Accuracy is ", accuracy_score(y_test, y_pred)*100)

print("Information gain Accuracy is ", accuracy_score(y_test, y_pred_en)*100)

#clf.feature_importances_
