import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ridge import MyRidge
from sklearn.model_selection import train_test_split

def load_data():
	try:
		data = pd.read_csv("data.csv")
	except:
		print("Invalid file error.")
		sys.exit()
	print("data shape:", data.shape)
	columns = data.columns.tolist()

	# Normalization
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	#return (normalized_data.values, columns[0], columns[1])
	return (data.values, columns[0], columns[1])

def normalization(data):
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data, data_min, data_max

def denormalization(normalized_data, data_min, data_max):
	denormalized_data = normalized_data * (data_max - data_min) + data_min
	return denormalized_data

def plot_scatter(data, feature, target):
	plt.scatter(data[:, 0], data[:, 1])
	plt.title(f'Scatter Plot: {feature} vs {target}')
	plt.xlabel(feature)
	plt.ylabel(target)
	plt.show()

def plot_scatters_for_normalization(data, normalized_data, denormalized_data, feature, target):
	fig, axes = plt.subplots(1, 3, figsize=(15, 8))
	axes[0].scatter(data[:, 0], data[:, 1])
	axes[0].set_title(f'Scatter Plot(data): {feature} vs {target}')
	axes[0].set_xlabel(feature)
	axes[0].set_ylabel(target)
	axes[0].grid()
	axes[1].scatter(normalized_data[:, 0], normalized_data[:, 1])
	axes[1].set_title(f'Scatter Plot(normalized): {feature} vs {target}')
	axes[1].set_xlabel(feature)
	axes[1].set_ylabel(target)
	axes[1].grid()
	axes[2].scatter(denormalized_data[:, 0], denormalized_data[:, 1])
	axes[2].set_title(f'Scatter Plot(denormalized): {feature} vs {target}')
	axes[2].set_xlabel(feature)
	axes[2].set_ylabel(target)
	axes[2].grid()
	plt.show()

def plot_scatter_with_prediction(data, x_test, y_pred, feature, target):
	plt.scatter(data[:, 0], data[:, 1], label='data')
	plt.plot(x_test, y_pred, c='orange', label='prediction line')
	plt.title(f'Scatter Plot: {feature} vs {target}')
	plt.xlabel(feature)
	plt.ylabel(target)
	plt.legend()
	plt.show()

def train(data, data_min, data_max):
	print(f"Starting training for linear regression...")
	x_train, x_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.2, random_state=42)
	thetas = np.zeros((2, 1))
	model = MyRidge(thetas, alpha=1e-2, max_iter=50000, lambda_=0.0)
	model.fit_(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
	print("thetas(original):", thetas)
	print("thetas(optimized):", model.thetas)
	return model

def get_prediction_data(data, data_min, data_max):
	x_train, x_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.2, random_state=42)
	y_pred = model.predict_(x_test.reshape(-1, 1))
	denormalized_x_test = denormalization(x_test.reshape(-1 ,1), data_min, data_max)
	denormalized_y_pred = denormalization(y_pred.reshape(-1, 1), data_min[-1], data_max[-1])

	# Sort the denormalized test data by the feature values
	sorted_indices = np.argsort(denormalized_x_test[:, 0])
	sorted_x_test = denormalized_x_test[sorted_indices, 0]
	sorted_predictions = denormalized_y_pred[sorted_indices]
	#print("prediction:", y_pred)
	#plot_scatter_with_prediction(denormalization(data, data_min, data_max), sorted_features, sorted_predictions, feature, target)
	return sorted_x_test, sorted_predictions

if __name__ == "__main__":
	# Load the data
	data, feature, target = load_data()

	# Normalization
	normalized_data, data_min, data_max = normalization(data)
	denormalized_data = denormalization(normalized_data, data_min, data_max)

	# Plot data scatters 
	#plot_scatter(data, feature, target)
	#plot_scatters_for_normalization(data, normalized_data, denormalized_data, feature, target)

	# Train the model on training set
	model = train(normalized_data, data_min, data_max)

	# Predict on test set
	x_test, predictions = get_prediction_data(normalized_data, data_min, data_max)

	# Plot the scatter and prediction line
	plot_scatter_with_prediction(data, x_test, predictions, feature, target)


