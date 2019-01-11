import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess
from visualize import visualize
from tester import test_classifier


#######################################################################
### Given Columns
''' 
PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Survived
'''

### Loading the Dataset
dataset = pd.read_csv('dataset/train.csv')


### Preprocess the data
feature_list = ['Fare', 'Sex_n', 'Age', 'Embarked_n']
features, labels = preprocess(dataset, feature_list, test_set=False)


### Visualizing Data
# visualize(dataset)


### Classification Model
clf = DTClassifier(min_samples_split=27)


### Train-Test Split
# X_train, X_val, y_train, y_val = \
# 		train_test_split(features, labels, test_size=0.2, random_state=42)

# clf.fit(X_train, y_train)
# print "Training Set Score:", clf.score(X_train, y_train)
# print "Validation Set Score:", clf.score(X_val, y_val)


### Cross Validation of Model

#test_classifier(clf, features, labels)


############################################################################
### Make predictions

test_set = pd.read_csv('dataset/test.csv')
test_features = preprocess(test_set, feature_list)

clf.fit(features, labels)
predictions = clf.predict(test_features)

output_df = pd.DataFrame()
output_df['PassengerId'] = test_set['PassengerId']
output_df['Survived'] = pd.Series(predictions)

with open('dataset/DTC_out.csv', 'w') as output_csv:
	output_df.to_csv(output_csv, sep=',', index=False)





