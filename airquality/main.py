import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

rawAir = pd.read_csv('./dataIN/AirQuality.csv',index_col=0)
rawCountries = pd.read_csv('./dataIN/CountryContinents.csv',index_col=0)
labels = list(rawAir.columns.values[1:])

#1
merged = rawAir.merge(rawCountries['Continent'],left_index=True,right_index=True)
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)
merged.set_index('Country')[labels].idxmax(axis=0).to_csv('./dataOUT/Requirement_1.csv',index_label=['Indicator'],header=['Country'])

#2
merged.set_index('Country').groupby('Continent').apply(lambda df: pd.Series({lab: df[lab].idxmax() for lab in labels})).to_csv('./dataOUT/Requirement_2.csv')

#3
x = StandardScaler().fit_transform(rawAir[labels])
HC = linkage(x, method='ward')
print(HC)
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

#4
plt.figure(figsize=(12, 12))
plt.title('Dendogram')
dendrogram(HC, labels=merged['Country'].values, leaf_rotation=45)
plt.axhline(t, c='r')
plt.show()

#5
cat = fcluster(HC, n - j, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]
merged['Cluster'] = clusters
merged[['Country','Cluster']].to_csv('./dataOUT/OptPart.csv')