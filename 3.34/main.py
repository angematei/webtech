import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

rawInd = pd.read_csv('./dataIN/Industrie.csv',index_col=0)
rawPop = pd.read_csv('./dataIN/PopulatieLocalitati.csv',index_col=0)
labels = list(rawInd.columns.values[1:])

#1
merged = rawInd.merge(rawPop[['Judet','Populatie']], left_index=True, right_index=True)
merged.fillna(np.mean(merged[labels], axis=0), inplace=True)
merged.set_index('Localitate', append=True).apply(lambda row: row[labels]/row['Populatie'], axis=1).to_csv('./dataOUT/Requirement_1.csv')

#2
merged_2 = merged[['Judet']+labels].groupby('Judet').sum()
merged_2['Turnover'] = merged_2.max(axis=1)
merged_2['Activity'] = merged_2.idxmax(axis=1)
merged_2[['Activity','Turnover']].to_csv('./dataOUT/Requirement_2.csv')

#3
rawProduction = pd.read_csv('./dataIN/DataSet_34.csv', index_col=0)
indexes = rawProduction.index
prodLab = rawProduction.columns[:4]
conLab = rawProduction.columns[4:]

x = pd.DataFrame(data=StandardScaler().fit_transform(rawProduction[prodLab]), index=indexes, columns=prodLab)
y = pd.DataFrame(data=StandardScaler().fit_transform(rawProduction[conLab]), index=indexes, columns=conLab)

x.to_csv('./dataOUT/Xstd.csv')
y.to_csv('./dataOUT/Ystd.csv')

#4
n, p = x.shape
q = y.shape[1]
m = min(p,q)
modelCCA = CCA(n_components=m)
modelCCA.fit(x,y)
z, u = modelCCA.transform(x,y)

ZLab = ['Z' + str(i+1) for i in range(z.shape[1])]
ULab = ['U' + str(i+1) for i in range(u.shape[1])]
pd.DataFrame(data=z, index=indexes, columns=ZLab).to_csv('./dataOUT/Xscore.csv')
pd.DataFrame(data=u, index=indexes, columns=ULab).to_csv('./dataOUT/Yscore.csv')

#5
Rxz = np.corrcoef(x, z[:,:m], rowvar=False)[:p, p:]
Ryu = np.corrcoef(y, u[:,:m], rowvar=False)[:q, q:]

pd.DataFrame(data=Rxz, index=ZLab, columns=prodLab).to_csv('./dataOUT/Rxz.csv')
pd.DataFrame(data=Ryu, index=ULab, columns=conLab).to_csv('./dataOUT/Ryu.csv')

#6
plt.figure(figsize=(7,7))
plt.title('Biplot (z1,z2) / (z2,u2)')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(z[:,0],z[:,1],c='r',label='X')
plt.scatter(u[:,0],u[:,1],c='b',label='Y')
plt.legend()
plt.show()