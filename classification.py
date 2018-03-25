#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# column names
column_names = ['age','workclass','fnlwgt','education','education-num',
                'marital-status','occupation','relationship','race','sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 
                'native-country', 'target']

# Read in the CSV file
data = pd.read_csv('adult_data.csv',  names = column_names, header=0)


# Remove the leading space in categorical features
cols = data.columns
num_cols = data._get_numeric_data().columns
cat_clos = list(set(cols) - set(num_cols))

# get rid of the space in the begining of categorical features
for cat_name in cat_clos: 
    data[cat_name] = data[cat_name].apply(str.strip)



# remove the missing values
def missing_to_NA(data):
    if data == '?':
        result = np.nan
    else:
        result = data
    return result

def remove_missing(data_frame):
    cols = data_frame.columns
    for column_name in cols[:-1]:
        data[column_name] = data[column_name].apply(missing_to_NA)
    return data

data = remove_missing(data)

# drop na rows
clean_data = data.dropna()

# Convert target features into 0 and 1 
def target_convert(word):
    if word == '<=50K':
        result = 0
    else:
        result = 1
    return result

clean_data['target'] = clean_data['target'].apply(target_convert)



###############################################################################
# data visulization
# correlation relations
cor = clean_data.corr()
plt.figure(figsize=(12,6))
sns.heatmap(cor, cmap='Blues', annot=True)

# 
clean_data['age'].hist(bins=90)
clean_data['fnlwgt'].hist(bins=100)
clean_data['education-num'].hist(bins=20)
clean_data['hours-per-week'].hist(bins=20)

sns.pairplot(clean_data,hue='target')

sns.jointplot('age', 'hours-per-week', clean_data)

sns.countplot(x='age', hue='target', data=clean_data)
plt.figure(figsize=(12,6))
sns.countplot(x='hours-per-week', hue='target', data=clean_data)

sns.kdeplot(clean_data['age'])
sns.kdeplot(clean_data['hours-per-week'])

###############################################################################


for cat_name in cat_clos: 
    clean_data[cat_name] = pd.Categorical(clean_data[cat_name])
    

for cat_name in cat_clos:
    print(cat_name + ':' + str(clean_data[cat_name].nunique()))

work = pd.get_dummies(clean_data['workclass'],drop_first=True)
education = pd.get_dummies(clean_data['education'],drop_first=True)
marital = pd.get_dummies(clean_data['marital-status'],drop_first=True)
occupation = pd.get_dummies(clean_data['occupation'],drop_first=True) 
relation = pd.get_dummies(clean_data['relationship'],drop_first=True)
race = pd.get_dummies(clean_data['race'],drop_first=True)
sex = pd.get_dummies(clean_data['sex'],drop_first=True)

# native country have too mancy different values, drop it
# concat the new data set
new_data = pd.concat([clean_data, work, education, marital, occupation, relation, race, sex], axis = 1)

cat_clos.remove('target')
new_data_copy = new_data.drop(cat_clos, axis=1)


# up to here, the data set is ready

# train test split
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, classification_report

X = new_data_copy.drop('target', axis=1)
y = new_data_copy['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# build machine learning model

accuracy = []
for n in range(1,51,2):
    classifier = KNeighborsClassifier(n_neighbors= n )
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    a = 1 - np.sum(pred != y_test) / len(y_test)
    accuracy.append(a)

plt.plot(range(1,51,2),accuracy, '--o')   

print(np.max(accuracy))

#print(confusion_matrix(y_test, pred))
#
#print(classification_report(y_test, pred))

### SVM 
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))















