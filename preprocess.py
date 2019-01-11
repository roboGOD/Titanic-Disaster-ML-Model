import pandas as pd

def preprocess(dataset, feature_list, return_labels=False):
	# Drop the Cabin Column
	dataset = dataset.drop(['Cabin'], axis=1)

	# Assign mean to the Age column NA values
	dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())


	# Convert Strings to numbers
	l = []
	for i in dataset['Sex']:
		if i == "male":
			l.append(0.)
		elif i == "female":
			l.append(1.)
		else:
			l.append(i)
	dataset['Sex_n'] = pd.Series(l)

	l = []
	for i in dataset['Embarked']:
		if i == "S":
			l.append(0.)
		elif i == "C":
			l.append(1.)
		elif i == "Q":
			l.append(2.)
		else:
			l.append(i)
	dataset['Embarked_n'] = pd.Series(l)


	# Drop all the NA values
	dataset = dataset.dropna()

	if return_labels:
		features = dataset[feature_list].values
		labels = dataset['Survived'].values
		return features, labels
	else:
		features = dataset[feature_list].values
		return features






