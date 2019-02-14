# Titanic Disaster Dataset ML Model
## Competition Description
>The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
>
>One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
>
>In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## Software Environment:
- Python version 2.7.15+
- Platform: (x86_64) 4.18.0-kali1-amd64

## External Dependencies and Libraries:
- pandas
- sklearn
- matplotlib

## Preprocessing
### Filling Missing Values
The 'Age' Column and 'Cabin' Column has significant number of missing values as shown by ```dataset.info()``` on combined train and test dataset.
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 12 columns):
PassengerId    1309 non-null int64
Survived       891 non-null float64
Pclass         1309 non-null int64
Name           1309 non-null object
Sex            1309 non-null object
Age            1046 non-null float64
SibSp          1309 non-null int64
Parch          1309 non-null int64
Ticket         1309 non-null object
Fare           1308 non-null float64
Cabin          295 non-null object
Embarked       1307 non-null object
dtypes: float64(3), int64(4), object(5)
memory usage: 122.8+ KB
None
```
The ```na``` values in 'Cabin' can be filled with "Unknown" as we don't know the cabin number. The missing values in 'Fare' can be filled with median as there is a single missing value. Similarly, 'Embarked' column's missing values can be filled with most common value i.e. ```mode``` of column.

We'll fill the missing values in the 'Age' Column by partitioning the data into groups on the basis of 'Title' Column. The 'Title' Column is extracted from 'Name' Column and median of the groups is used to fill 'Age' column.

The 'Title' Column is further mapped into more general titles. So we have following groups as shown by ```dataset.Title.value_counts()```.
```
Mr        761
Miss      264
Mrs       201
Master     61
Dr          8
Rev         8
Col         4
Major       2
Name: Title, dtype: int64
``` 

### Engineering new feature columns
There are two new engineered columns 'Title' and 'FamilySize'. We discussed 'Title' column above. 'FamilySize' is computed as below:
```
FamilySize = SibSp + Parch + 1
```

### Creating Bins
After filling ```na``` values, 'Age' is partitioned into 4 bins using ```pandas.qcut()``` as the survival patterns are based on age groups rather than exact value. Similarly, 'Fare' is partitioned into 5 bins.

### Encoding the categorical columns into integer values
```LabelEncoder()``` the following columns are transformed and encoded to integer values.

```
Cabin => Cabin_Code
Embarked => Embarked_Code
Sex => Sex_Code
Title => Title_Code
AgeBin => AgeBin_Code
FareBin => FareBin_Code
``` 

### Columns used to learn the model
- Pclass 
- Cabin_Code 
- Embarked_Code 
- Sex_Code 
- Title_Code 
- AgeBin_Code
- FareBin_Code
- FamilySize



## Training the model

## Acknowledgements
