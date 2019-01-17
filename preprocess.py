import pandas as pd

def preprocess(dataset):
	# Drop the Cabin Column
	# dataset = dataset.drop(['Cabin'], axis=1)
	dataset['Cabin'] = dataset['Cabin'].fillna('Missing')
	l = []
	for i in dataset['Cabin']:
		l.append(i[0])
	dataset['Cabin_n'] = pd.Series(l)

	# print dataset['Cabin_n'].unique()

	### Fill in missing values
	# Assign mean to the Age column NA values
	dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

	dataset['Embarked'] = dataset['Embarked'].fillna('S')



	# Convert Strings to numbers
	# l = []
	# for i in dataset['Sex']:
	# 	if i == "male":
	# 		l.append(0.)
	# 	elif i == "female":
	# 		l.append(1.)
	# 	else:
	# 		l.append(i)
	# dataset['Sex_n'] = pd.Series(l)

	dataset.loc[dataset['Sex'] == 'male', "Sex"] = 0.
	dataset.loc[dataset['Sex'] == 'female', "Sex"] = 1.


	# l = []
	# for i in dataset['Embarked']:
	# 	if i == "S":
	# 		l.append(0.)
	# 	elif i == "C":
	# 		l.append(1.)
	# 	elif i == "Q":
	# 		l.append(2.)
	# 	else:
	# 		l.append(i)
	# dataset['Embarked_n'] = pd.Series(l)

	dataset.loc[dataset['Embarked'] == 'S', "Embarked"] = 0.
	dataset.loc[dataset['Embarked'] == 'C', "Embarked"] = 1.
	dataset.loc[dataset['Embarked'] == 'Q', "Embarked"] = 2.

	ls = ['M','C','E','G','D','A','B','F','T']
	for i,j in enumerate(ls):
		dataset.loc[dataset['Cabin_n'] == j, "Cabin_n"] = float(i)	

	return dataset



def split_features_labels(dataset, feature_list, test_set=True):
	if not test_set:
		features = dataset[feature_list].values
		labels = dataset['Survived'].values
		return features, labels
	else:
		features = dataset[feature_list].values
		return features


	






