import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

rawAir = pd.read_csv('./dataIN/AirQualityCountries.csv',index_col=0)
rawCodes = pd.read_csv('./dataIN/CountryCodes.csv',index_col=0)
labels = list(rawAir.columns.values[1:])

#1
merged = rawAir.merge(rawCodes,left_index=True,right_index=True)
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)
# merged[['Country']+labels].set_index('Country').idxmax(axis=0).to_csv('./dataOUT/Requirement_1.csv')
merged.set_index('Country')[labels].idxmax(axis=0).to_csv('./dataOUT/Requirement_1.csv',index_label='Indicator',header=['Country'])

#2
merged.set_index('Country').groupby('Continent').apply(lambda df: pd.Series({lab: df[lab].idxmax() for lab in labels})).to_csv('./dataOUT/Requirement_2.csv')

#3
x = StandardScaler().fit_transform(rawAir[labels])
HC = linkage(x.T, method='average', metric='correlation')
print(HC)

#4
n = HC.shape[0]
dist_1 = HC[1:n, 2]
dist_2 = HC[0:n - 1, 2]
diff = dist_1 - dist_2
j = np.argmax(diff)
t = (HC[j, 2] + HC[j + 1, 2]) / 2

print('junction with max df:', j)
print('threshold:', np.round(t, 2))

#5
plt.figure(figsize=(12, 12))
plt.title('Dendogram')
dendrogram(HC, labels=labels, leaf_rotation=45)
plt.axhline(t, c='r')
plt.show()

#6
cat = fcluster(HC, n - j, criterion='maxclust')
clusters = ['C' + str(i) for i in cat]
pd.DataFrame(data={'Cluster': clusters}, index=labels).to_csv('./dataOUT/OptPart.csv')