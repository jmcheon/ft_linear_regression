import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import imageio.v2 as imageio
from train import fit_, load_data, normalization, denormalization, denormalize_thetas
from predict import predict_

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

		new_thetas = fit_(partial_x, partial_y, thetas, alpha=3e-1)
		denormalized_thetas = denormalize_thetas(new_thetas, data_max, data_min)
		denormalized_partial_x = denormalization(partial_x.reshape(-1 ,1), data_min[0], data_max[0])
		denormalized_y_pred = predict_(denormalized_partial_x, denormalized_thetas)

		fig = plot_scatter_with_prediction(data, denormalized_partial_x, denormalized_y_pred, feature, target)
		plt.title(f'theta:{denormalized_thetas}')
		
		# Call save to create temporary image file.
		temp_img_file = f'temp_{i}.png'
		fig.savefig(temp_img_file)
		plt.close()
		
		images.append(imageio.imread(temp_img_file))
	imageio.mimsave(filename, images, duration=0.5)
	return new_thetas

if __name__ == "__main__":
	# Load the data
	data, feature, target = load_data("data.csv")

	# Normalization
	normalized_data, data_min, data_max = normalization(data)
	#print(f"data_min:{data_min}, data_max:{data_max}")
	denormalized_data = denormalization(normalized_data, data_min, data_max)

	# Plot data scatters 
	plot_scatter(data, feature, target)
	# plot_scatters_for_normalization(data, normalized_data, denormalized_data, feature, target)

	# Load the thetas 
	try:
		thetas = pd.read_csv('model.csv').values
	except:
		print("Invalid file error.")
		sys.exit()
	print(f"denormalized thetas: {thetas}, {thetas.shape}")

	# Predict
	y_pred = predict_(data[:, 0].reshape(-1, 1), thetas)

	# Plot the scatter and prediction line
	plt = plot_scatter_with_prediction(data, data[:, 0], y_pred, feature, target)
	plt.show()

	create_animated_gif(data, feature, target, num_steps=len(data[:, 0]))
