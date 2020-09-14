import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as  pd
import os

# importando os dados 
data = pd.read_csv("./datasets_6_class.csv")
# vamos dar uma olhada nos 5 primeiros dados
print(data.head())


# conhecendo os dados
print(data.shape)
print(data.info())
sns.pairplot(data = data, hue = "Star type")
plt.show()
sns.distplot(data["Temperature (K)"])
plt.show()
sns.distplot(data['Luminosity(L/Lo)'])
plt.show()

ax = sns.countplot(data["Star color"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

print(data["Star type"].value_counts())