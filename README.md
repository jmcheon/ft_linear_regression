# ft_linear_regression - An introduction to machine learning

>_**Summary: In this project, you will implement your first machine learning algorithm.**_

<br>

## Usage
1. Prediction before training
```python
python3 predict.py
```
- output
```
Type a Km to estimate price: 3421
Estimated price for km: 3421 is 0.
```
2. Train
- It generates `model.csv` which contains optimized theta values
```python
python3 train.py
```

3. Prediction after training
```python
python3 predict.py
```
- output
```
Type a Km to estimate price: 3421
Estimated price for km: 3421 is 8426.
```

4. Evaluation
- Evaluate the model using `r2 score`
```python
python3 evaluate.py
```

6. Visualization
```python
python3 plot.py
```

## Visualization

| Before Training | After Training |
|:---------------:|:--------------:|
|![temp_1](https://github.com/jmcheon/ft_linear_regression/assets/40683323/b76255fa-ea88-4426-9534-f5997782b1f0)|![temp_24](https://github.com/jmcheon/ft_linear_regression/assets/40683323/bc800057-35c3-4c48-8e71-640956e34774)|




<br>

### Animated Training Process
![scatter_regression](https://github.com/jmcheon/ft_linear_regression/assets/40683323/020074bc-03f0-463a-9863-f0080e9225f3)