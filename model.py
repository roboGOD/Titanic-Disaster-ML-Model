import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from preprocess import preprocess, split_features_labels
from visualize import visualize
from tester import test_classifier

style.use('ggplot')

#######################################################################
### Given Columns
''' 
PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Survived
'''

### Loading the Dataset
dataset = pd.read_csv('dataset/train.csv')


### Preprocess the data
dataset = preprocess(dataset)
dataset = dataset.dropna()

feature_list = ['Fare', 'Sex', 'Age', 'Embarked', 'Pclass', 'SibSp', 'Cabin_n']
features, labels = split_features_labels(dataset, feature_list, test_set=False)

### Visualizing Data
# visualize(dataset)


### Train-Test Split
X_train, X_val, y_train, y_val = \
		train_test_split(features, labels, test_size=0.2, random_state=42)


### Classification Model

base_est = DTClassifier(min_samples_split=14, max_depth=11, random_state=42)

######################## SVC ###############################
### Finding best value for C as 6.5
# samp_vals = [6+x*0.1 for x in range(1,40)]
# accuracies = []
# for i in samp_vals:
# 	clf = SVC(C=i, random_state=42)
# 	clf.fit(X_train, y_train)
# 	accuracies.append(clf.score(X_val, y_val))

# plt.plot(samp_vals, accuracies, '-b')
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.show()

### Finding best value for gamma as 0.0023
# samp_vals = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
# samp_vals = [x*0.0001 for x in range(1,100)]
# accuracies = []
# for i in samp_vals:
# 	clf = SVC(C=6.5, gamma=i, random_state=42)
# 	clf.fit(X_train, y_train)
# 	accuracies.append(clf.score(X_val, y_val))

# plt.plot(samp_vals, accuracies, '-b')
# plt.xlabel('gamma')
# plt.ylabel('Accuracy')
# plt.show()



# clf = RandomForestClassifier(n_estimators = 5, min_samples_split=8, max_depth=21,  random_state=42)
# clf = KNeighborsClassifier(n_neighbors = 18, weights = 'distance', p=1)
# clf = AdaBoostClassifier(base_estimator=base_est, n_estimators=20, random_state=42)
# clf = GaussianNB()
clf = SVC(C=6.5, gamma=0.0023, random_state=42)
clf_ = clf.fit(X_train, y_train)



# plt.bar(feature_list, clf.feature_importances_)
# plt.show()


print "Training Set Score:", clf.score(X_train, y_train)
print "Validation Set Score:", clf.score(X_val, y_val)


### Cross Validation of Model

test_classifier(clf, features, labels, folds=100)


############################################################################
### Make predictions

test_set = pd.read_csv('dataset/test.csv')
test_set = preprocess(test_set)
test_features = split_features_labels(test_set, feature_list)

clf.fit(features, labels)
predictions = clf.predict(test_features)

output_df = pd.DataFrame()
output_df['PassengerId'] = test_set['PassengerId']
output_df['Survived'] = pd.Series(predictions)

with open('dataset/_out.csv', 'w') as output_csv:
	output_df.to_csv(output_csv, sep=',', index=False)

print "Done!"



