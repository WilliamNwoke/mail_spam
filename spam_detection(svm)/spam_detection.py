
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm

##Step1: load dataset
dataframe =pd.read_csv("spam.csv")
print(dataframe.head())
##Step2: Split in to Training and Test Data
x = dataframe["EmailText"]
y = dataframe["Label"]

x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

##Step3: Extract Features
cv = CountVectorizer()
features = cv.fit_transform(x_train)

##Step4: Build a model
tuned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3, 1e-4],'C':[1,10,100,1000]}

model = GridSearchCV(svm.SVC(),tuned_parameters)
model.fit(features, y_train)

print(model.best_params_)

##Step5: Test Accuracy

features_test = cv.transform(x_test)

print("Accuracy of the model is:",model.score(features_test, y_test) )