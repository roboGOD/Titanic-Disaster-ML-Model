import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(dataset):

	###  Preprocessing to do
	# Do the grouping of data for filling the missing age values
	# Create bins of Age and Fare
	# Engineer a feature Family Size = SibSp + Parch + 1
	# Include the Title Column
	# Find people in same family by grouping by Last Name
	# Do something about the categorical features
	# Scale all the features

	le = LabelEncoder()
	# print dataset.info()


	## Preprocessing 'Cabin' Column
	dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
	dataset['Cabin_Code'] = le.fit_transform(dataset['Cabin'].apply(lambda x: x[0]))
	dataset = pd.get_dummies(dataset, columns=['Cabin_Code'], prefix="Cabin")


	### Preprocessing 'Embarked' Column
	dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode().values[0])
	dataset['Embarked_Code'] = le.fit_transform(dataset['Embarked'])
	dataset = pd.get_dummies(dataset, columns=['Embarked_Code'], prefix="Em")


	### Preprocessing 'Sex' Column
	dataset['Sex_Code'] = le.fit_transform(dataset['Sex'])


	### Preprocessing 'Age' Column
	## Creating a 'Title' Column for grouping
	dataset['Title'] = dataset['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
	uniqueTitles = dataset['Title'].unique()
	
	## Generalize the titles 
	mapTitles = {'Mr':'Mr', 'Mrs':'Mrs', 'Miss':'Miss', 'Master':'Master', 'Don':'Mr', 
	'Rev':'Rev', 'Dr':'Dr', 'Mme':'Mrs', 'Ms':'Miss', 'Major':'Major', 'Lady':'Mrs', 'Sir':'Mr', 
	'Mlle':'Miss', 'Col':'Col', 'Capt':'Mr', 'the Countess':'Mrs', 'Jonkheer':'Mr', 'Dona':'Mrs'}

	dataset['Title'] = dataset['Title'].map(mapTitles)
	# print dataset.Title.value_counts()
	dataset['Title_Code'] = le.fit_transform(dataset['Title'])
	dataset = pd.get_dummies(dataset, columns=['Title_Code'], prefix="Title")
	
	dataset['Age'] = dataset.groupby(['Title', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
	dataset['AgeBin'] = pd.qcut(dataset['Age'], 4)
	dataset['AgeBin_Code'] = le.fit_transform(dataset['AgeBin'])

	### Preprocessing 'Fare' Column
	dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
	dataset['FareBin'] = pd.qcut(dataset['Fare'], 5, labels=["1", "2", "3", "4", "5"])
	dataset['FareBin_Code'] = le.fit_transform(dataset['FareBin'])

	### Creating 'FamilySize' Feature
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
	dataset['Single'] = dataset['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
	dataset['SmallFam'] = dataset['FamilySize'].apply(lambda x: 1 if x == 2 else 0)
	dataset['MedFam'] = dataset['FamilySize'].apply(lambda x: 1 if (x == 3 or x == 4) else 0)
	dataset['BigFam'] = dataset['FamilySize'].apply(lambda x: 1 if x >= 5 else 0)


	### Preprocessing 'Ticket' Column
	tCol = dataset['Ticket'].apply(lambda x: x.replace(".", "").replace("/", "").strip().split(" ")[0])
	dataset['TicketPrefix'] = tCol.apply(lambda x: "X" if x.isdigit() else x)
	dataset['TPrefix_Code'] = le.fit_transform(dataset['TicketPrefix'])
	dataset = pd.get_dummies(dataset, columns=['TPrefix_Code'], prefix="TPre")

	### Drop unnecessary Columns
	columns_drop = ['Age', 'AgeBin', 'Fare', 'FareBin', 'SibSp', 'Parch', 'Title',
	 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'TicketPrefix', 'FamilySize']
	dataset = dataset.drop(columns_drop, axis=1)
	
	return dataset.astype('float64')