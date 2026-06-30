import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawAlc = pd.read_csv('./dataIN/alcohol.csv',index_col=0)
rawCodes = pd.read_csv('./dataIN/CoduriTariExtins.csv',index_col=0)
labels = list(rawAlc.columns.values[1:])

#1
merged = rawAlc.merge(rawCodes['Continent'],left_index=True,right_index=True)
merged.fillna(np.mean(rawAlc[labels],axis=0),inplace=True)
merged.apply(lambda row: np.average(row[labels]),axis=1).sort_values(ascending=False).to_csv('./dataOUT/Requirement_1.csv',header=['Average Consumption'])

#2
merged[['Continent']+labels].groupby('Continent').mean().idxmax(axis=1).to_csv('./dataOUT/Requirement_2.csv',header=['Year'])

#3
x = StandardScaler().fit_transform(merged[labels])
HC = linkage(x, method='ward')
print(HC)

#4
cat = fcluster(HC, 5, criterion='maxclust')
clusters =  ['C' + str(i) for i in cat]
merged['Clusters'] = clusters
merged[['Clusters']].to_csv('./dataOUT/3.csv')

#5
pca = PCA()
C = pca.fit_transform(x)
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans_labels = kmeans.fit_predict(C)
plt.figure(figsize=(8, 6))
plt.scatter(C[:, 0], C[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-means clustering")
plt.show()