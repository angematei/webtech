import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rawInd = pd.read_csv('./dataIN/GlobalIndicatorsPerCapita_2021.csv',index_col=0)
rawCountries = pd.read_csv('./dataIN/CountryContinents.csv',index_col=0)
labels = list(rawInd.columns.values[1:])
indexes = list(rawInd.index.values)

#1
merged = rawInd.merge(rawCountries[['Continent']], left_index=True, right_index=True)
merged.fillna(np.mean(merged[labels],axis=0),inplace=True)
merged[['Country'] + labels[-7:]].set_index('Country',append=True).sum(axis=1).to_csv('./dataOUT/Requirement_1.csv', header=['ValoareAdaugata'])

#2
merged[['Continent']+labels].groupby('Continent').apply(lambda df: pd.Series({ind: np.round(np.std(df[ind]) / np.mean(df[ind]) * 100, 2) for ind in labels})).to_csv('./dataOUT/Requirement_2.csv')

#3
x = StandardScaler().fit_transform(merged[labels])
pca = PCA()
C = pca.fit_transform(x)
alpha = pca.explained_variance_
print(alpha)

#4
scores = C / np.sqrt(alpha)
pd.DataFrame(data=np.round(scores, 2), index=indexes, columns=labels).to_csv('./dataOUT/Scores.csv')

#5
plt.figure(figsize=(9, 9))
plt.title('Scores')
plt.scatter(scores[:, 0], scores[:, 1])
plt.show()