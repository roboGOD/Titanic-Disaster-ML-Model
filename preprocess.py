import pandas as pd

def preprocess(dataset):
	# Drop the Cabin Column
	dataset = dataset.drop(['Cabin'], axis=1)

	# Assign mean to the Age column NA values
	# dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
	surv_data = dataset[dataset['Survived'] == 1]
	died_data = dataset[dataset['Survived'] == 0]

	surv_data['Age'] = surv_data['Age'].fillna(surv_data['Age'].mean())
	died_data['Age'] = died_data['Age'].fillna(died_data['Age'].mean())
	dataset = surv_data.append(died_data).sort_index()

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

	return dataset



def split_features_labels(dataset, feature_list, test_set=True):
	if not test_set:
		# Drop all the NA values
		dataset = dataset.dropna()
		features = dataset[feature_list].values
		labels = dataset['Survived'].values
		return features, labels
	else:
		# Fill NA values with 0.
		dataset = dataset.fillna(0.)
		features = dataset[feature_list].values
		return features


	






