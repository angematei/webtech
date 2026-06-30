import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawNat = pd.read_csv('./dataIN/NatLocMovements.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/PopulationLoc.csv', index_col=0)
labels = list(rawNat.columns.values[1:])

#1
merged = rawNat.merge(rawPop, left_index=True, right_index=True)[['City','Population','CountyCode'] + labels]
# merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
# merged.groupby('City').apply(lambda row: row['DeceasedUnder1Year']/row['LiveBirths']).to_csv('./dataOUT/Requirement_1.csv')
merged['MortalityRate'] = merged['DeceasedUnder1Year']/merged['LiveBirths']*100
merged[['City','MortalityRate']].to_csv('./dataOUT/Requirement_1.csv')

#2
merged.set_index(['City','CountyCode']).apply(lambda row: row[labels]/row['Population']*1000, axis=1).reset_index(1).groupby('CountyCode').apply(lambda df: pd.Series({lab: df[lab].idxmax() for lab in labels})).to_csv('./dataOUT/Requirement_2.csv')

#3
rawData = pd.read_csv('./dataIN/DataSet_83.csv', index_col=0)
labels = list(rawData.columns.values[1:])
rows = list(rawData.index.values)

x = StandardScaler().fit_transform(rawData[labels])
cov = np.cov(x, rowvar=False)
pd.DataFrame(np.round(cov, 2), index=labels, columns=labels).to_csv('./dataOUT/StdCov.csv')

#4
pca = PCA()
C = pca.fit_transform(x)

pd.DataFrame(np.round(C, 2), index=rows, columns=['C' + str(i+1) for i in range(C.shape[1])]) \
.to_csv('./dataOUT/PrinComp.csv')

# 5
alpha = pca.explained_variance_

plt.figure(figsize=(8, 8))
plt.title('Variance explained by the principal components')
Xindex = ['C' + str(k + 1) for k in range(len(alpha))]
plt.plot(Xindex, alpha, 'bo-')
plt.axhline(1, color='r')
plt.show()

# 6
a = pca.components_.T
rxc = a * np.sqrt(alpha)

plt.figure(figsize=(8, 8))
plt.title('Factor loadings')
T = [t for t in np.arange(0, np.pi * 2, 0.01)]
X = [np.cos(t) for t in T]
Y = [np.sin(t) for t in T]
plt.plot(X, Y)
plt.axhline(0, c='g')
plt.axvline(0, c='g')
plt.scatter(rxc[:, 0], rxc[:, 1])
plt.show()