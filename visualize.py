from matplotlib import pyplot as plt
from matplotlib import style


def visualize(dataset):
	surv_data = dataset[dataset['Survived'] == 1]
	died_data = dataset[dataset['Survived'] == 0]

	style.use('ggplot')

	# plt.scatter(surv_data['Fare'].values, surv_data['Age'].values, label='Survived', color='g')
	# plt.scatter(died_data['Fare'].values, died_data['Age'].values, label='Died', color='r')

	# plt.hist(surv_data['Fare'], label='Survived', color='g')
	# plt.hist(died_data['Fare'], label='Died', color='r')

	plt.subplot(1,2,1)
	plt.hist(surv_data['Age'], label='Survived', color='g')
	plt.xlabel('Age')
	plt.ylabel('Frequency')
	plt.title('Survived')

	plt.subplot(1,2,2)
	plt.hist(died_data['Age'], label='Died', color='r')
	plt.xlabel('Age')
	plt.ylabel('Frequency')
	plt.title('Died')

	plt.show()















