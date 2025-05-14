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
california_housing=fetch_california_housing()
data=pd.DataFrame(california_housing.data,columns=california_housing.feature_names)
corr_matrix=data.corr()
print(corr_matrix)
plt.figure()
sns.heatmap(corr_matrix,annot=True,fmt='.2f')
plt.title("corerealtaion")
plt.show()
plt.figure()
sns.pairplot(data,kind="scatter",diag_kind='kde')
plt.show()

3)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris=load_iris()
x=iris.data
y=iris.target
label_names=iris.target_names
df=pd.DataFrame(x,columns=iris.feature_names)
df
pca=PCA(n_components=2)
principal_components=pca.fit_transform(x)
df_pca=pd.DataFrame(data=principal_components,
columns=['Principal Component 1','Principal Component 2'])
df_pca['Target']=y
plt.figure()
colors=['y','b','g']
for i, label in enumerate(np.unique(y)):
  plt.scatter(df_pca[df_pca['Target']==label]['Principal Component 1'],
            df_pca[df_pca['Target']==label]['Principal Component 2'],
            label=label_names[label],
            color=colors[i])
plt.title('pca')
plt.xlabel('1')
plt.ylabel('2')
plt.legend()
plt.show()


4)
import csv
a=[]
with open('enjoysport.csv','r') as csvfile:
  for row in csv.reader(csvfile):
    a.append(row)

print("totla instances",len(a))
num_attributes=len(a[0])-1
hypothesis=['0']*num_attributes
print('initial one',hypothesis)

for i in range(len(a)):
  if a[i][num_attributes]=='yes':
    for j in range(num_attributes):
      if hypothesis[j]=='0':
        hypothesis[j]=a[i][j]
      elif hypothesis[j]!=a[i][j]:
        hypothesis[j]='?'
  print("hypo for trainisning {}".format(i+1),hypothesis)
print(hypothesis)

5)
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data=np.random.rand(100)
labels=['class1' if x<=0.5 else'class2' for x in data[:50]]

def euc(x1,x2):
  return abs(x1-x2)

def knn_classifier(train_data,train_labels,test_point,k):
  distance=[(euc(test_point,train_data[i]),train_labels[i])
             for i in range(len(train_data))]
  distance.sort(key=lambda x:x[0])
  k_nearest_neighbour=distance[:k]
  k_nearest_labels=[label for _, label in k_nearest_neighbour]
  return Counter(k_nearest_labels).most_common(1)[0][0]

train_data=data[:50]
test_data=data[50:]
train_labels=labels

k_values=[1,2,3,4,5,20,30]
result={}
for k in k_values:
  print(f'results of k={k}')
  classified_labels=[knn_classifier(train_data,train_labels,test_point,k) for test_point in test_data]
  results[k]=classified_labels
  for i,label in enumerate(classified_labels,start=51):
    print(f'point x{i} values={test_data[i-51]:.4f} label={label}')
print("done")

num_k=len(k_values)
rows=(num_k+2)//3
cols=3
plt.figure(figsize=(15,5*rows))
for idx,k in enumerate(k_values):
  classified_labels=results[k]
  class1point=[test_data[i] for i in range(len(test_data)) if classified_labels[i]=='class1']
  class2point=[test_data[i] for i in range(len(test_data)) if classified_labels[i]=='class2']

  plt.subplot(rows,cols,idx+1)
  plt.scatter(train_data,[0]*len(train_data),
              c=['blue' if label=='class1' else 'red' for label in train_labels],
              label="Training data",marker="o")
  plt.scatter(class1point,[1]*len(class1point),c='blue',label='class1 test',marker='x')
  plt.scatter(class2point,[1]*len(class2point),c='red',label='class2 test',marker='x')

  plt.title(f'knnr results for k={k}')
  plt.xlabel('data points')
  plt.ylabel('classification level')
  plt.legend()
  plt.grid(True)
plt.tight_layout()
plt.show()
