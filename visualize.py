from matplotlib import pyplot as plt
from matplotlib import style


def visualize(dataset):
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















