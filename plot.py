import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.model_selection import train_test_split
from fit import train_model, fit_, load_data, normalization, denormalization
from prediction import predict_

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
	#plt.show()
	return plt

def create_animated_gif(data, feature, target, num_steps, filename='scatter_regression.gif'):
	normalized_data, data_min, data_max = normalization(data)
	images = []
	print("Creating an animated gif image...")
	for i in range(1, num_steps + 1):
		partial_x = normalized_data[:i, 0].reshape(-1, 1)
		partial_y = normalized_data[:i, 1].reshape(-1, 1)
		thetas = np.zeros((2, 1)).reshape(-1, 1)

		new_thetas = fit_(partial_x, partial_y, thetas, alpha=1e-2, max_iter=5000)
		y_pred = predict_(partial_x, new_thetas)

		denormalized_partial_x = denormalization(partial_x.reshape(-1 ,1), data_min[0], data_max[0])
		denormalized_y_pred = denormalization(y_pred.reshape(-1, 1), data_min[-1], data_max[-1])

		fig = plot_scatter_with_prediction(data, denormalized_partial_x, denormalized_y_pred, feature, target)
		plt.title(f'theta:{new_thetas}')
		
		# Call save to create temporary image file.
		temp_img_file = f'temp_{i}.png'
		fig.savefig(temp_img_file)
		plt.close()
		
		images.append(imageio.imread(temp_img_file))
	imageio.mimsave(filename, images, duration=0.5)
	return new_thetas

def r2_score(data, thetas):
	x = data[:, 0]
	y = data[:, 1]
	ssm = np.sum(y - np.average(y))

if __name__ == "__main__":
	# Load the data
	data, feature, target = load_data()

	# Normalization
	normalized_data, data_min, data_max = normalization(data)
	#print(f"data_min:{data_min}, data_max:{data_max}")
	denormalized_data = denormalization(normalized_data, data_min, data_max)

	# Plot data scatters 
	#plot_scatter(data, feature, target)
	#plot_scatters_for_normalization(data, normalized_data, denormalized_data, feature, target)

	# Train the model on training set
	thetas = train_model()

	# Predict on test set
	x_train, x_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.2, random_state=42)
	y_pred = predict_(x_test.reshape(-1, 1), thetas)

	# Plot the scatter and prediction line
	plt = plot_scatter_with_prediction(data, x_test, y_pred, feature, target)
	plt.show()
	create_animated_gif(data, feature, target, num_steps=len(data[:, 0]))
