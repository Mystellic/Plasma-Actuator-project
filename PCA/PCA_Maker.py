# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:31:13 2020

@author: Robin
"""

import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

values = pandas.read_csv('KLM_Aircraft_Specs_Choice.csv',sep=';',names=['Range','Passengers','MTOW','target'])
titles = ['Range','Passengers','MTOW']


# Separating out the features
x = values.loc[:, titles].values

# Separating out the target
y = values.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
components = pca.fit_transform(x)
principalValues = pd.DataFrame(data = components
             , columns = ['principal component 1', 'principal component 2'])


#final DF
finalValues = pd.concat([principalValues, df[['target']]], axis = 1)

#visualizing the PCA

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Airbus A330-200', 'Airbus A330-300', 'Boeing 737-700','Boeing 737-800','Boeing 737-900','Boeing 747-400','Boeing 777-200ER','Boeing 777-300ER','Boeing 787-9','Embraer 175']
colors = ['r', 'r', 'b','b','b','b','b','b','b','g']
for target, color in zip(targets,colors):
    indicesToKeep = finalValues['target'] == target
    ax.scatter(finalValues.loc[indicesToKeep, 'principal component 1']
               , finalValues.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()









