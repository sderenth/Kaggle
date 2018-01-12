# Titanic Data Set

# Kaggle Competition
# Started: 1.8.18
# Last updated:

# Goal: Predict who lived and who died.


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# set working directory
os.chdir('C:\Users\Sean\Kaggle\Titanic')

# read data into DataFrames
df = pd.DataFrame(pd.read_csv('train.csv'))

# simple model for baseline.
df_simple = df[['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']]
train_simple, test_simple = train_test_split(df_simple, test_size=0.4)

x_train_simple = train_simple[['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare']]
y_train_simple = train_simple['Survived']

x_test_simple = test_simple[['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare']]
y_test_simple = test_simple['Survived']

# model_simple = RandomForestClassifier(random_state=1)
# model_simple.fit(x_train_simple, y_train_simple)
# score = model_simple.score(x_test_simple, y_test_simple)
# print 'Model: simple, Accuracy: ', score
# print ''

# Acc: .666666666667


# # How many trees?
# tree_nums = range(1000, 5000, 1000)
# scores = []
# 
# for tree_num in tree_nums:
#     model = RandomForestClassifier(tree_num, random_state=1, n_jobs=-1)
#     model.fit(x_train_simple, y_train_simple)
#     
#     print tree_num, 'trees'
#     score = model.score(x_test_simple, y_test_simple)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, tree_nums).plot()
# plt.show()

# Tree = 1000. No benefit after.



# # Max depth of trees?
# tree_depths = range(1, 7, 1)
# scores = []
# 
# for tree_depth in tree_depths:
#     model = RandomForestClassifier(1000, max_depth=tree_depth, random_state=1, n_jobs=-1)
#     model.fit(x_train_simple, y_train_simple)
#     
#     print tree_depth, 'tree splits'
#     score = model.score(x_test_simple, y_test_simple)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, tree_depths).plot()
# plt.show()
# 
# # Barely seems to matter. Let's go 4, seemed to be the best one most of the time.


# # Minimum sample required in node to split again?
# min_sams = range(2, 40, 4)
# scores = []
# 
# for min_sam in min_sams:
#     model = RandomForestClassifier(1000, min_samples_split=min_sam, random_state=1, n_jobs=-1)
#     model.fit(x_train_simple, y_train_simple)
#     
#     print min_sam, 'Min node # for split'
#     score = model.score(x_test_simple, y_test_simple)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, min_sams).plot()
# plt.show()
# 
# # Leave default


# # Minimum sample leaf?
# leaves = range(1, 40, 4)
# scores = []
# 
# for leaf in leaves:
#     model = RandomForestClassifier(1000, max_depth=4, min_samples_leaf=leaf, random_state=1, n_jobs=-1)
#     model.fit(x_train_simple, y_train_simple)
#     
#     print leaf, 'Min node # for split'
#     score = model.score(x_test_simple, y_test_simple)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, leaves).plot()
# plt.show()
# 
# # Leave default (1)


# Final Simple Model (baseline)
model_simple = RandomForestClassifier(1000, random_state=1, n_jobs=-1)
model_simple.fit(x_train_simple, y_train_simple)
score = model_simple.score(x_test_simple, y_test_simple)
print 'Model Simple.  Accuracy: ', score



### Data Cleaning/Feature Engineering ###

# Dummify Pclass
dummy_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
df = pd.concat([df, dummy_pclass], axis=1)

# Get title from 'Name'.
df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0]
df.drop('Name', axis=1, inplace=True)

dummy_titles = pd.get_dummies(df['Title'], prefix='Title')
df = pd.concat([df, dummy_titles], axis=1)
df.drop('Title', axis=1, inplace=True)

# Dummify SibSp
dummy_sibsp = pd.get_dummies(df['SibSp'], prefix='SibSp')
df = pd.concat([df, dummy_sibsp], axis=1)

# Dummify Parch
dummy_parch = pd.get_dummies(df['Parch'], prefix='Parch')
df = pd.concat([df, dummy_parch], axis=1)

# Replace 'Age' NaN, with 'Age' mean.
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

# Replace 'Embarked' Nan, with 'S'
df['Embarked'] = df['Embarked'].fillna('S')

dummy_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, dummy_embarked], axis=1)
df.drop('Embarked', axis=1, inplace=True)

# Dummy variables for Sex
dummy_sex = pd.get_dummies(df['Sex'])
df = pd.concat([df, dummy_sex], axis=1)
df.drop(['Sex', 'male'], axis=1, inplace=True)

# Separate Ticket Number and Prefix text into two columns.
df['Ticket Number'] = df['Ticket'].str.split(' ').str[-1]
df.replace('LINE', np.nan, inplace=True)
df['Ticket Number'] = pd.to_numeric(df['Ticket Number'])
df['Ticket Number'] = df['Ticket Number'].fillna(df['Ticket Number'].mean())

df['Ticket Prefix'] = np.where(df['Ticket'].str.split(' ').str[0].str.isdigit() == False, df['Ticket'].str.split(' ').str[0].str.replace('.', ''), 'None')
dummy_tic_pre = pd.get_dummies(df['Ticket Prefix'], prefix='TickPref')
df = pd.concat([df, dummy_tic_pre], axis=1)
df.drop('Ticket Prefix', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)




### Model 2 ###

df = df.drop('Cabin', axis=1)

train, test = train_test_split(df, test_size=0.4)

x_train = train.drop('Survived', axis=1)
y_train = train['Survived']

x_test = test.drop('Survived', axis=1)
y_test = test['Survived']


model = RandomForestClassifier(1000, random_state=1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print 'Model V2 baseline.  Accuracy: ', score

# # Variable Strength
# # Graph showing the relative importance of each variable on the final prediction of the model.
# var_str = pd.Series(model.feature_importances_, index=x_train.columns)
# var_str.sort_values(inplace=True)
# var_str.plot(kind='barh', grid=True)
# plt.show()


# # How many trees?
# tree_nums = range(400, 1000, 50)
# scores = []
# 
# for tree_num in tree_nums:
#     model = RandomForestClassifier(tree_num, random_state=1, n_jobs=-1)
#     model.fit(x_train, y_train)
#     
#     print tree_num, 'trees'
#     score = model.score(x_test, y_test)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, tree_nums).plot()
# plt.show()
# 
# # Tree = 1000. No benefit after.
# 
# 
# 
# # Max depth of trees?
# tree_depths = range(1, 80, 4)
# scores = []
# 
# for tree_depth in tree_depths:
#     model = RandomForestClassifier(750, max_depth=tree_depth, random_state=1, n_jobs=-1)
#     model.fit(x_train_simple, y_train_simple)
#     
#     print tree_depth, 'tree splits'
#     score = model.score(x_test_simple, y_test_simple)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, tree_depths).plot()
# plt.show()

# Use None, or default.


# # Minimum sample required in node to split again?
# min_sams = range(2, 60, 4)
# scores = []
# 
# for min_sam in min_sams:
#     model = RandomForestClassifier(1000, min_samples_split=min_sam, random_state=1, n_jobs=-1)
#     model.fit(x_train, y_train)
#     
#     print min_sam, 'Min node # for split'
#     score = model.score(x_test, y_test)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, min_sams).plot()
# plt.show()
# 
# # Leave default (2)


# # Minimum sample leaf?
# leaves = range(1, 15, 1)
# scores = []
# 
# for leaf in leaves:
#     model = RandomForestClassifier(1000, min_samples_leaf=leaf, random_state=1, n_jobs=-1)
#     model.fit(x_train, y_train)
#     
#     print leaf, 'min leaf size'
#     score = model.score(x_test, y_test)
#     print 'Accuracy: ', score
#     scores.append(score)    
#     print ''
#     
# pd.Series(scores, leaves).plot()
# plt.show()
# 
# # Leaf min = 2






### Submission on Test Data ###

# read data into DataFrames
test_df = pd.DataFrame(pd.read_csv('test.csv'))

# Dummify Pclass
dummy_pclass = pd.get_dummies(test_df['Pclass'], prefix='Pclass')
test_df = pd.concat([test_df, dummy_pclass], axis=1)

# Get title from 'Name'.
test_df['Title'] = test_df['Name'].str.split(',').str[1].str.split('.').str[0]
test_df.drop('Name', axis=1, inplace=True)

dummy_titles = pd.get_dummies(test_df['Title'], prefix='Title')
test_df = pd.concat([test_df, dummy_titles], axis=1)
test_df.drop('Title', axis=1, inplace=True)

# Dummify SibSp
dummy_sibsp = pd.get_dummies(test_df['SibSp'], prefix='SibSp')
test_df = pd.concat([test_df, dummy_sibsp], axis=1)

# Dummify Parch
dummy_parch = pd.get_dummies(test_df['Parch'], prefix='Parch')
test_df = pd.concat([test_df, dummy_parch], axis=1)

# Replace 'Age' NaN, with 'Age' mean.
age_mean = test_df['Age'].mean()
test_df['Age'] = test_df['Age'].fillna(age_mean)

# Replace 'Embarked' Nan, with 'S'
test_df['Embarked'] = test_df['Embarked'].fillna('S')

dummy_embarked = pd.get_dummies(test_df['Embarked'], prefix='Embarked')
test_df = pd.concat([test_df, dummy_embarked], axis=1)
test_df.drop('Embarked', axis=1, inplace=True)

# Dummy variables for Sex
dummy_sex = pd.get_dummies(test_df['Sex'])
test_df = pd.concat([test_df, dummy_sex], axis=1)
test_df.drop(['Sex', 'male'], axis=1, inplace=True)

# Separate Ticket Number and Prefix text into two columns.
test_df['Ticket Number'] = test_df['Ticket'].str.split(' ').str[-1]
test_df.replace('LINE', np.nan, inplace=True)
test_df['Ticket Number'] = pd.to_numeric(test_df['Ticket Number'])
test_df['Ticket Number'] = test_df['Ticket Number'].fillna(test_df['Ticket Number'].mean())

test_df['Ticket Prefix'] = np.where(test_df['Ticket'].str.split(' ').str[0].str.isdigit() == False, test_df['Ticket'].str.split(' ').str[0].str.replace('.', ''), 'None')
dummy_tic_pre = pd.get_dummies(test_df['Ticket Prefix'], prefix='TickPref')
test_df = pd.concat([test_df, dummy_tic_pre], axis=1)
test_df.drop('Ticket Prefix', axis=1, inplace=True)
test_df.drop('Ticket', axis=1, inplace=True)

# Fare, fillna(mean)
fare_mean = test_df['Fare'].mean()
test_df['Fare'] = test_df['Fare'].fillna(fare_mean)

# Drop 'Cabin'
test_df = test_df.drop('Cabin', axis=1)


# Add missing columns
missing_test_columns = list(set(x_train.columns.tolist()) - set(test_df.columns.tolist()))
missing_train_columns = list(set(test_df.columns.tolist()) - set(x_train.columns.tolist()))


for column in missing_test_columns:
    test_df[column] = 0

for column in missing_train_columns:
    x_train[column] = 0
    x_test[column] = 0

# x_train.drop(missing_test_columns, axis=1, inplace=True)
# x_test.drop(missing_test_columns, axis=1, inplace=True)
# 
# test_df.drop(missing_train_columns, axis=1, inplace=True)

print len(x_train.columns)
print len(x_test.columns)
print len(test_df.columns)  

x_train.to_csv('titanic_x_train.csv', index=False)
x_test.to_csv('titanic_x_test.csv', index=False)
y_train.to_csv('titanic_y_train.csv', index=False)
y_test.to_csv('titanic_y_test.csv', index=False)
test_df.to_csv('titanic_test_set.csv', index=False)

# Final Model v 2
model = RandomForestClassifier(1000, min_samples_leaf=1, random_state=1, n_jobs=-1)
model.fit(x_train, y_train)
    
score = model.score(x_test, y_test)
print 'Model V2. Accuracy: ', score


# Get preditction for Test set.
test_preds = pd.DataFrame(model.predict(test_df))

test_preds['PassengerId'] = test_preds.index + 892
test_preds['Survived'] = test_preds[test_preds.columns[0]]
test_preds.drop(test_preds.columns[0], axis=1, inplace=True)
