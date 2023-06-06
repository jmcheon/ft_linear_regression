import numpy as np
from train import train_model

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

if __name__ == "__main__":
	# Train the model
	thetas = train_model()
	print(f"denormalized thetas: {thetas}, {thetas.shape}")

	# Predict
	km = input("\nType a Km to estimate price: ")
	if km.isdigit():
		estimated_price = int(predict_(np.array(float(km)).reshape(-1, 1), thetas))
		print(f"Estimated price for km: {km} is {estimated_price}.")
	else:
		print(f"Invalid input: {km}, positive integer value required.")
