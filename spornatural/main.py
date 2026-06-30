import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawRate = pd.read_csv('./dataIN/Rata.csv',index_col=0)
rawCodes = pd.read_csv('./dataIN/CoduriTariExtins.csv',index_col=0)
labels = list(rawRate.columns.values[1:])

#1
merged = rawRate.merge(rawCodes[['Continent']],left_index=True,right_index=True)
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)
merged[merged['RS']<np.average(merged['RS'])][['Country_Name','RS']].sort_values('RS',ascending=False).to_csv('./dataOUT/Requirement_1.csv')

#2
merged.groupby('Continent').apply(lambda df: pd.Series({ind: df[ind].idxmax() for ind in labels})).to_csv('./dataOUT/Requirement_2.csv')

#3
x = StandardScaler().fit_transform(merged[labels])
pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
pve = pca.explained_variance_ratio_
var_cum = np.cumsum(alpha)
pve_cum = np.cumsum(pve)

pd.DataFrame(data={'Component variance': alpha, 'Cum variance': var_cum, 'Explained variance percentage': pve, 'Cum percentage': pve_cum}).to_csv('./dataOUT/Varianta.csv')

#4
plt.figure(figsize=(10, 10))
plt.title('Variance explained by components')
labels = ['C' + str(i + 1) for i in range(len(alpha))]
plt.plot(labels, alpha, 'bo-')
plt.axhline(1, c='r')
plt.show()

#5
a = pca.components_.T
Rxc = a * np.sqrt(alpha)
communalities = np.cumsum(Rxc * Rxc, axis=1)
communalities_df = pd.DataFrame(data=communalities, index=labels, columns=['C' + str(i + 1) for i in range(communalities.shape[1])])

plt.figure(figsize=(9, 9))
plt.title('Correlogram')
sb.heatmap(communalities_df, vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()