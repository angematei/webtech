import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

rawNat = pd.read_csv('./dataIN/NatLocMovement.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/PopulationLoc.csv', index_col=0)
labels = list(rawNat.columns.values[1:])

#1
merged = rawNat.merge(rawPop[['County','Population']], left_index=True, right_index=True)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
merged.reset_index().groupby('County').sum().apply(lambda row: row['LiveBirths']/row['Population']*1000 - row['Deceased']/row['Population']*1000, axis=1).to_csv('./dataOUT/Requirement_1.csv')

#2
merged.set_index(['Locality','County']).apply(lambda row: row[labels]/row['Population']*1000, axis=1).reset_index(1).groupby('County').apply(lambda df: pd.Series({lab: df[lab].idxmax() for lab in labels})).to_csv('./dataOUT/Requirement_2.csv')

#3
rawHealth = pd.read_csv('./dataIN/DataSet_65.csv', index_col=1).drop(columns='Id')
x = StandardScaler().fit_transform(rawHealth)
pd.DataFrame(x, columns=rawHealth.columns.values).to_csv('./dataOUT/Xstd.csv')

HC = linkage(x, method='ward')
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
dendrogram(HC, labels=rawHealth.index.values, leaf_rotation=45)
plt.axhline(t, c='r')
plt.show()