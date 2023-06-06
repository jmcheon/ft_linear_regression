import numpy as np
import pandas as pd

def load_data():
	try:
		data = pd.read_csv("data.csv")
	except:
		print("Invalid file error.")
		sys.exit()
	print("data shape:", data.shape)
	columns = data.columns.tolist()

	return (data.values, columns[0], columns[1])

def normalization(data):
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data, data_min, data_max

def denormalization(normalized_data, data_min, data_max):
	x = normalized_data * (data_max - data_min)
	denormalized_data = normalized_data * (data_max - data_min) + data_min
	return denormalized_data

def denormalize_thetas(thetas, data_max, data_min):
	# Recover the slope of the line
	slope = thetas[1] * (data_max[1] - data_min[1]) / (data_max[0] - data_min[0])
	# Recover the intercept of the line
	intercept = thetas[0] * (data_max[1] - data_min[1]) + data_min[1] - slope * data_min[0]
	denormalized_thetas = np.array([intercept, slope]).reshape(-1, 1)
	return denormalized_thetas

def fit_(x, y, thetas, alpha):
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

	# Weights to update: alpha * mean((y_hat - y) * x) 
	# Bias to update: alpha * mean(y_hat - y)
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	new_theta = np.copy(thetas.astype("float64"))
	new_mse = 0.0
	i = 0
	while True:
		y_hat = X.dot(new_theta)
		# Compute gradient descent
		b_grad = np.mean(y_hat - y)
		w_grad = np.mean((y_hat - y) * x)
		grad = np.array([b_grad, w_grad]).reshape(-1, 1)
		mse = np.sum(np.square(y_hat - y)) / len(y)
		gain = mse - new_mse
		i += 1
		#print("mse:", mse, "new:", new_mse, "i:", i)
		if gain == 0:
			break
		new_mse = mse
	        # Handle invalid values in the gradient
		if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
			#print("Warning: Invalid values encountered in the gradient. Skipping update.")
			continue
		# Update new_theta
		new_theta -= (alpha * grad)
	thetas = new_theta
	return thetas

def train(data):
	print(f"Starting training for linear regression...")
	thetas = np.zeros((2, 1))
	x = data[:, 0]
	y = data[:, 1]
	new_thetas = fit_(x.reshape(-1, 1), y.reshape(-1, 1), thetas, alpha=3e-1)
	print("thetas(original):", thetas)
	print("thetas(optimized):", new_thetas)
	return new_thetas

def train_model():
	# Load the data
	data, feature, target = load_data()

	# Normalization
	data, data_min, data_max = normalization(data)
	#print(f"data_min:{data_min}, data_max:{data_max}")
	x = data[:, 0]
	y = data[:, 1]

	print(f"Starting training for linear regression...")
	thetas = np.zeros((2, 1))
	new_thetas = fit_(x.reshape(-1, 1), y.reshape(-1, 1), thetas, alpha=3e-1)
	print("thetas(original):", thetas, thetas.shape)
	print("thetas(optimized):", new_thetas, new_thetas.shape)

	return denormalize_thetas(new_thetas, data_max, data_min)

if __name__ == "__main__":
	# Load the data
	data, feature, target = load_data()

	# Normalization
	normalized_data, data_min, data_max = normalization(data)

	# Train the model on training set
	thetas = train(normalized_data)
