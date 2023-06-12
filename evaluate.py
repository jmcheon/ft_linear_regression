import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score
from train import load_data

def r2_score_(data, thetas):
	#print(f"thetas: {thetas}")
	x = data[:, 0].reshape(-1, 1)
	y = data[:, 1].reshape(-1, 1)
	X = np.hstack((np.ones((x.shape[0], 1)), x))

	y_pred = X.dot(thetas).reshape(-1, 1)
	y_mean = np.mean(y)

	ssr = np.sum((y - y_pred) ** 2)
	sst = np.sum((y - y_mean) ** 2)

	r2_score_value = 1 - (ssr / sst)
	#print(r2_score(data[:, 1], y_pred))
	return r2_score_value

if __name__ == "__main__":
	# Load the data
	data, feature, target = load_data("data.csv")
	try:
		thetas = pd.read_csv('model.csv').values
	except:
		print("Invalid file error.")
		sys.exit()

	#print(f"denormalized thetas: {thetas}, {thetas.shape}")
	print(f"R2 score: {r2_score_(data, thetas):.2f}")
