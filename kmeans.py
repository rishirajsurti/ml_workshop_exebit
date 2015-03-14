import numpy as np;
import matplotlib.pyplot as plt;
from sklearn import datasets, cluster, metrics

iris = datasets.load_iris()
labels = iris.target
iris_x = iris.data[:]
iris_x = iris_x[:, 0:2];
kmeans = cluster.KMeans(n_clusters = 3);
kmeans.fit(iris_x);

plt.figure()
plt.scatter(iris_x[:,0], iris_x[:,1], c = labels, marker = 'o', label = 'actual classes');
plt.scatter(iris_x[:,0], iris_x[:,1], c = kmeans.labels, marker = '+', label = 'assigned classes');
plt.legend(loc = 'upper right');
plt.show();