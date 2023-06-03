import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def fit_(x, y, thetas, alpha, max_iter):
	for v in [x, y, thetas]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if y.ndim == 1:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)
		return None
	
	if x.shape[0] != y.shape[0]:
		print(f"Invalid input: x, y matrices should be compatible.")
		return None

	if thetas.ndim == 1 and thetas.size == x.shape[1] + 1:
		thetas = thetas.reshape(x.shape[1] + 1, 1)
	elif not (thetas.ndim == 2 and thetas.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of {thetas}", thetas.shape)
		return None

	if not isinstance(alpha, float) or alpha <= 0:
		print(f"Invalid input: argument alpha of positive float type required")	
		return False

	if not isinstance(max_iter, int) or max_iter <= 0:
		print(f"Invalid input: argument max_iter of positive integer type required")	
		return False 

	# Weights to update: alpha * mean((y_hat - y) * x) 
	# Bias to update: alpha * mean(y_hat - y)
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	new_theta = np.copy(thetas.astype("float64"))
	for _ in range(max_iter):
		y_hat = X.dot(new_theta)
		# Compute gradient descent
		w_grad = np.mean((y_hat - y) * x)
		b_grad = np.mean(y_hat - y)
		grad = np.array([b_grad, w_grad]).reshape(-1, 1)
	        # Handle invalid values in the gradient
		if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
			#print("Warning: Invalid values encountered in the gradient. Skipping update.")
			continue
		# Update new_theta
		new_theta -= (alpha * grad)
	thetas = new_theta
	return thetas

def predict_(x, thetas):
	for v in [x, thetas]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not x.ndim == 2:
		print("Invalid input: wrong shape of x", x.shape)
		return None

	if thetas.ndim == 1 and thetas.size == x.shape[1] + 1:
		thetas = thetas.reshape(x.shape[1] + 1, 1)
	elif not (thetas.ndim == 2 and thetas.shape == (x.shape[1] + 1, 1)):
		print(f"p Invalid input: wrong shape of {thetas}", thetas.shape)
		return None
	
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = X.dot(thetas)
	return np.array(y_hat)

def train(data, data_min, data_max):
	print(f"Starting training for linear regression...")
	x_train, x_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.2, random_state=42)
	thetas = np.zeros((2, 1))
	new_thetas = fit_(x_train.reshape(-1, 1), y_train.reshape(-1, 1), thetas, alpha=1e-2, max_iter=5000)
	print("thetas(original):", thetas)
	print("thetas(optimized):", new_thetas)
	return new_thetas

if __name__ == "__main__":
	# Load the data
	data, feature, target = load_data()

	# Normalization
	normalized_data, data_min, data_max = normalization(data)
	denormalized_data = denormalization(normalized_data, data_min, data_max)

	# Train the model on training set
	thetas = train(normalized_data, data_min, data_max)

	# Predict
	km = input("\nType a Km for estimated price for it: ")
	normalized_km = (float(km) - data_min[0]) / (data_max[0] - data_min[0])
	estimated_price = predict_(np.array(normalized_km).reshape(-1, 1), thetas) 
	estimated_price = int(denormalization(estimated_price, data_min[-1], data_max[-1]))
	print(f"Estimated price for km: {km} is {estimated_price}.")
