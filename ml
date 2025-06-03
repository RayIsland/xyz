5)
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]

def euclidean_distance(x1, x2):
    return abs(x1 - x2)

def knn_classifier(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i])
                 for i in range(len(train_data))]
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]
    k_nearest_labels = [label for _, label in k_nearest_neighbors]
    return Counter(k_nearest_labels).most_common(1)[0][0]

train_data = data[:50]
train_labels = labels
test_data = data[50:]

k_values = [1, 2, 3, 4, 5, 20, 30]

print("--- k-Nearest Neighbors Classification ---")
print("Training dataset: First 50 points labeled based on the rule (x <= 0.5 -> Class1, x > 0.5 -> Class2)")
print("Testing dataset: Remaining 50 points to be classified\n")

results = {}

for k in k_values:
    print(f"Results for k = {k}:")
    classified_labels = [knn_classifier(train_data, train_labels, test_point, k) for test_point in test_data]
    results[k] = classified_labels
    for i, label in enumerate(classified_labels, start=51):
        print(f"Point x{i} (value: {test_data[i - 51]:.4f}) is classified as {label}")
    print("\n")

print("Classification complete.\n")

num_k = len(k_values)
rows = (num_k + 2) // 3 
cols = 3  
plt.figure(figsize=(15, 5 * rows)) 

for idx, k in enumerate(k_values):
    classified_labels = results[k]
    class1_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class1"]
    class2_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class2"]

    plt.subplot(rows, cols, idx + 1)
    plt.scatter(train_data, [0] * len(train_data),
                c=["blue" if label == "Class1" else "red" for label in train_labels],
                label="Training Data", marker="o")
    plt.scatter(class1_points, [1] * len(class1_points), c="blue", label="Class1 (Test)", marker="x")
    plt.scatter(class2_points, [1] * len(class2_points), c="red", label="Class2 (Test)", marker="x")

    plt.title(f"k-NN Results for k = {k}")
    plt.xlabel("Data Points")
    plt.ylabel("Classification Level")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

1)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()

data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
numerical_features = data.select_dtypes(include=[np.number]).columns
print(numerical_features)

data.hist(bins=30,figsize=(15,10))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns, 1):
  plt.subplot(3, 3, i)
  sns.boxplot(y=data[column])
  plt.title(f'Box Plot of {column}')
plt.tight_layout()
plt.show()

print("Outliers Detection:\n")
for feature in numerical_features:
  Q1 = data[feature].quantile(0.25)
  Q3 = data[feature].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
  print(f"\t{feature}: {len(outliers)} outliers\t")

2)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

corr_matrix = data.corr()
print(corr_matrix)

plt.figure()
sns.heatmap(corr_matrix, annot=True,fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

plt.figure()
sns.pairplot(data, kind='scatter',diag_kind='kde')
plt.show()

3)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target 
label_names = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

df_pca = pd.DataFrame(data=principal_components,
columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y

plt.figure()
colors = ['y', 'b', 'g']
for i, label in enumerate(np.unique(y)):
  plt.scatter(df_pca[df_pca['Target'] == label]['Principal Component 1'],
            df_pca[df_pca['Target'] == label]['Principal Component 2'],
            label=label_names[label],
            color=colors[i])
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

4)
import csv
a = []
with open('enjoysport.csv', 'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)

print("\nThe total number of training instances are:", len(a))
num_attribute = len(a[0]) - 1
hypothesis = ['0'] * num_attribute
print("\nThe initial hypothesis is:")
print(hypothesis)

for i in range(len(a)):
    if a[i][num_attribute] == 'yes':
        for j in range(num_attribute):
            if hypothesis[j] == '0':
                hypothesis[j] = a[i][j]
            elif hypothesis[j] != a[i][j]:
                hypothesis[j] = '?'
    print("\nThe hypothesis for the training instance {} is:\n".format(i + 1), hypothesis)
print("\nThe Maximally specific hypothesis for the training instance is:")
print(hypothesis)
