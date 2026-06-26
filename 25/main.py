import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rawBudget = pd.read_csv('./dataIN/Buget.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/LocPopulation.csv', index_col=0)

labels = list(rawBudget.columns.values[1:])

#1
merged = rawBudget.merge(rawPop[['County','Population']], left_index=True, right_index=True)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
merged.reset_index().set_index(['Siruta','Locality']).apply(lambda row: row[labels]/row['Population'], axis=1).to_csv('./dataOUT/Requirement_1.csv')

#2
expenses = list(rawBudget.columns.values[6:])
merged[['County'] + expenses].groupby('County').sum().apply(lambda row: row[expenses]/sum(row[expenses])*100, axis=1).to_csv('./dataOUT/Requirement_2.csv')

#3
rawData = pd.read_csv('./dataIN/DataSet_25.csv', index_col=0)
labels = list(rawData.columns.values[1:])
rows = list(rawData.index.values)

x = StandardScaler().fit_transform(rawData[labels])

cov = np.cov(x, rowvar=False)
pd.DataFrame(np.round(cov, 2), index=labels, columns=labels).to_csv('./dataOUT/StdCov.csv')

#4
pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
print(alpha)

#5
plt.figure(figsize=(8,8))
plt.title('Variance explained by the principal components')
Xindex = ['C' + str(k+1) for k in range(len(alpha))]
plt.plot(Xindex, alpha, 'bo-')
plt.axhline(1, color='r')
plt.show()

#6
a = pca.components_.T
rxc = a * np.sqrt(alpha)

plt.figure(figsize=(8,8))
plt.title('Factor loadings')
T = [t for t in np.arange(0, np.pi * 2, 0.01)]
X = [np.cos(t) for t in T]
Y = [np.sin(t) for t in T]
plt.plot(X,Y)
plt.axhline(0,c='g')
plt.axvline(0,c='g')
plt.scatter(rxc[:,0],rxc[:,1])
plt.show()

rxc_df = pd.DataFrame(data=rxc, index=labels, columns=['C' + str(i+1) for i in range(rxc.shape[1])])
plt.figure(figsize=(10,10))
plt.title('Correlogram')
sb.heatmap(rxc_df, vmin=-1, vmax=1, annot=True, cmap='bwr')
plt.show()