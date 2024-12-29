import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

group_colors = np.array([
    [224, 240, 183], [187, 227, 99], [135, 161, 75],
    [84, 100, 48], [108, 99, 68]
    ])

scaler = MinMaxScaler()
group_colors_normalized = scaler.fit_transform(group_colors)

model = OneClassSVM(kernel="rbf", gamma="auto")
model.fit(group_colors_normalized)


color_list = [64, 130, 109]
new_color = np.array([color_list])  
new_color_normalized = scaler.transform(new_color)

decision_score = model.decision_function(new_color_normalized)[0]

min_score = model.decision_function(group_colors_normalized).min()
max_score = model.decision_function(group_colors_normalized).max()
probability = (decision_score - min_score) / (max_score - min_score)

print(f"Fit Probability: {probability:.2f}")