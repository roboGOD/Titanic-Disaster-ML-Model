import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

#######################################################################
### Loading the Dataset
dataset = pd.read_csv('dataset/train.csv')

### Given Columns
''' 
PassengerId
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
Survived
'''

#######################################################################
### Preprocess the data

# Drop the Cabin Column
nd_dataset = dataset.drop(['Cabin'], axis=1)

# Assign mean to the Age column NA values
nd_dataset['Age'] = nd_dataset['Age'].fillna(nd_dataset['Age'].mean())


# Convert Strings to numbers
l = []
for i in nd_dataset['Sex']:
	if i == "male":
		l.append(0.)
	elif i == "female":
		l.append(1.)
	else:
		l.append(i)
nd_dataset['Sex_n'] = pd.Series(l)

l = []
for i in nd_dataset['Embarked']:
	if i == "S":
		l.append(0.)
	elif i == "C":
		l.append(1.)
	elif i == "Q":
		l.append(2.)
	else:
		l.append(i)
nd_dataset['Embarked_n'] = pd.Series(l)


# Drop all the NA values
nd_dataset = nd_dataset.dropna()


# Modify the actual dataset
dataset = nd_dataset

'''
#######################################################################
### Visualizing Data
surv_data = dataset[dataset['Survived'] == 1]
died_data = dataset[dataset['Survived'] == 0]

style.use('fivethirtyeight')
#plt.scatter(surv_data['PassengerId'].values, surv_data['Age'].values, label='Survived', color='g')
#plt.scatter(died_data['PassengerId'].values, died_data['Age'].values, label='Died', color='r')

#plt.hist(surv_data['Fare'], label='Survived', color='g')
#plt.hist(died_data['Fare'], label='Died', color='r')

plt.subplot(1,2,1)
plt.hist(surv_data['Age'], label='Survived', color='g')
plt.xlabel('Age', size=10)
plt.ylabel('Frequency', size=10)
plt.xticks(size=8)
plt.yticks(size=8)
plt.title('Survived', size=10)

plt.subplot(1,2,2)
plt.hist(died_data['Age'], label='Died', color='r')
plt.xlabel('Age', size=10)
plt.ylabel('Frequency', size=10)
plt.xticks(size=8)
plt.yticks(size=8)
plt.title('Died', size=10)

plt.show()
'''


#######################################################################
### Train-test Split

feature_list = ['Fare', 'Sex_n', 'Age', 'Embarked_n']
labels = dataset['Survived'].values

X_train, X_val, y_train, y_val = \
		train_test_split(dataset[feature_list].values, labels, test_size=0.2, random_state=42)


### Classification Model
clf = AdaBoostClassifier(n_estimators = 25)
clf.fit(X_train, y_train)
print clf.score(X_val, y_val)
