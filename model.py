import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from preprocess import preprocess
from tester import test_classifier
# from visualize import visualize


#######################################################################
### Given Columns
''' 
PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Survived
'''

### Engineered Columns
'''
AgeBin_Code, FareBin_Code, Embarked_Code, Sex_Code, FamilySize, Cabin_Code, TPrefix_Code
'''

### Loading the datasets
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
df_full = df_train.append(df_test, sort=False, ignore_index=True)

### Preprocess the data
df_full = preprocess(df_full)
df_train = df_full[:891]
df_test = df_full[891:]
df_test.index -= 891

### Scale the features
scl = StandardScaler()
features = scl.fit_transform(df_train.drop(['Survived', 'PassengerId'], axis=1).values)
labels = df_train['Survived'].values


### Train-Test Split
X_train, X_val, y_train, y_val = \
		train_test_split(features, labels, test_size=0.2, random_state=42)

### Classification Model
### After Tuning Random Forest:
## n_estimators: 10, min_samples_split: 20, min_samples_leaf: 3, max_features: 'log2' 

# param_list = {'n_estimators':[x for x in range(4,60,3)],
# 			'min_samples_split':[x for x in range(2,25,2)],
# 			'min_samples_leaf':[x for x in range(1, 10, 2)],
# 			'max_depth':[x for x in range(1, 25, 2)],
# 			}

# rfc = RandomForestClassifier(max_features='log2', random_state=42)
# clf = GridSearchCV(rfc, param_list, cv=5, verbose=3, n_jobs=-1)
# clf_ = clf.fit(features, labels)
# print clf.best_score_
# print clf.best_estimator_

# print "Training Set Score:", clf.score(X_train, y_train)
# print "Validation Set Score:", clf.score(X_val, y_val)

### Cross Validation of Model
# test_classifier(clf.best_estimator_, features, labels, folds=100)

############################################################################
### Make predictions

# test_features = scl.fit_transform(df_test.drop(['Survived', 'PassengerId'], axis=1).values)

# clf.best_estimator_.fit(features, labels)
# predictions = clf.best_estimator_.predict(test_features)

# output_df = pd.DataFrame({'PassengerId':df_test['PassengerId'],
# 						  'Survived':pd.Series(predictions)})
# output_df = output_df.astype('Int64')

# with open('dataset/_out.csv', 'w') as output_csv:
# 	output_df.to_csv(output_csv, sep=',', index=False)

# print "Done!"



