# import numpy as np
# from sklearn.svm import OneClassSVM
# from sklearn.preprocessing import MinMaxScaler

# group_colors = np.array([
#     [243,239,232], [187,165,145], [250,236,195],
#     [51,102,89], [14,28,79]
#     ])

# scaler = MinMaxScaler()
# group_colors_normalized = scaler.fit_transform(group_colors)

# model = OneClassSVM(kernel="rbf", gamma="auto")
# model.fit(group_colors_normalized)


# color_list = [255,217,227]
# new_color = np.array([color_list])  
# new_color_normalized = scaler.transform(new_color)

# decision_score = model.decision_function(new_color_normalized)[0]

# print(f"Decision score {decision_score}")

# min_score = model.decision_function(group_colors_normalized).min()
# max_score = model.decision_function(group_colors_normalized).max()
# probability = (decision_score - min_score) / (max_score - min_score)

# print(f"Fit Probability: {probability:.2f}")


# import numpy as np
# from sklearn.cluster import KMeans
# from PIL import Image

# img = Image.open('/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top/cn55717004.jpg')
# img = img.resize((img.width // 10, img.height // 10))  # Resize to speed up processing
# img_array = np.array(img)

# pixels = img_array.reshape(-1, 3)

# kmeans = KMeans(n_clusters=2)
# kmeans.fit(pixels)

# dominant_color = kmeans.cluster_centers_[0]
# dominant_color = tuple(map(int, dominant_color))  # Convert to integer

# print(f'Dominant color (RGB): {dominant_color}')


import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# Load the image
img = Image.open('/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top/cn55717004.jpg')
img = img.convert('RGB')  # Ensure the image is in RGB format

# Get pixel data
pixels = np.array(img).reshape(-1, 3)

# Count occurrences of each color
color_counts = Counter(tuple(pixel) for pixel in pixels)

# Get most common colors (top 10 for example)
common_colors = color_counts.most_common(10)

# Print top 10 common colors and their counts
for color, count in common_colors:
    print(f'Color: {color}, Count: {count}')

# Plot histogram of color occurrences (just showing a small subset of colors)
colors, counts = zip(*common_colors)
colors = np.array(colors)

# Convert RGB to a format suitable for plotting (e.g., hexadecimal)