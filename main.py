import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Reading csv files
dftrain = pd.read_csv('Phising_Training_Dataset.csv')
dftest = pd.read_csv('Phising_Testing_Dataset.csv')
#Seperating Result Column to train model based on it to be predicted
train_y = dftrain.pop('Result')
#train_test_split
X_train, X_test, y_train, y_test = train_test_split(dftrain, train_y, test_size=0.3, random_state=101)

#training model
param_grid = {'n_estimators':[100,200,300,400,500,600,700,800,1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101),param_grid,verbose=3)
grid_cv.fit(X_train,y_train)
print(grid_cv.best_params_)

# checking the prediction
prediction = grid_cv.predict(dftrain)
print(f"Prediction:- {prediction[2000]}, Actual:- {train_y[2000]} ")
print(accuracy_score(train_y, prediction))

print(classification_report(train_y,prediction))
print('\n')
print(confusion_matrix(train_y,prediction))
print('\n')
print('Acuuracy is --> ',accuracy_score(train_y,prediction)*100)
print('\n')

#predicted the test dataset
prdic = grid_cv.predict(dftest)

#creating a dataframe of key and predicted result for test dataset
keys = dftest['key']
values = pd.DataFrame(prdic)
frames = [keys,values]
result = pd.concat(frames, axis=1)
result.columns= ['key','Result']


# converting from dataframe to csv file
result.to_csv('Submission.csv',index=False)

##Representing Two different result values using pie chart
count = result['Result'].value_counts()
legit = count[1]
phis = count[-1]
dt = {'Task':[legit,phis]}
wpp_df = pd.DataFrame(dt,columns=['Task'],index = ['Legit','Phishing'])
wpp_df.plot.pie(y='Task',figsize=(5, 5),autopct='%1.1f%%', startangle=90)
plt.show()
