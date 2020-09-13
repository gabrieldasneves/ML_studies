import pandas as pd
import matplotlib.pyplot as plt
# adiquirindo os dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)



# mudando a nomenclatura das colunas e trocando alguns valores
a_renomear = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
 }

dados = dados.rename(columns = a_renomear)
print("dados:")
print(dados.head())

trocar = {
    1:0,
    0:1
}

dados['finalizado'] = dados.nao_finalizado.map(trocar) # isso vai criar uma nova coluna
print(dados.head())

# plotando alguns valores
import seaborn as sns

sns.scatterplot(x="horas_esperadas", y = "preco", data=dados)
sns.scatterplot(x="horas_esperadas", y = "preco", hue="finalizado",data=dados)

x = dados[['horas_esperadas','preco']]
y = dados[["finalizado"]]

# fazendo a classificação
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

SEED = 5
np.random.seed(SEED)
raw_treino_x, raw_teste_x, treino_y,teste_y = train_test_split(x,y,random_state = SEED, test_size=0.25,stratify=y)
scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC() #criando
modelo.fit(treino_x,treino_y) #treinando
previsoes = modelo.predict(teste_x) #prevendo

acuracia = accuracy_score(teste_y, previsoes)*100
print('a acuracia foi de')
print( acuracia)

# precisamos ter um algoritmo de bas (baseline) para ver se essa acuracia é boa ou não
# ele sempre vai ser o pior caso que podemos ter de acuracia

previsoes_de_base = np.ones(540)
acuracia_baseline = accuracy_score(teste_y, previsoes_de_base)*100
print("previsoes de base: ")
print(acuracia_baseline)

data_x = teste_x[:,0]
data_y = teste_x[:,1]
x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()
pixels = 100
eixo_x= np.arange(x_min,x_max,(x_max-x_min)/pixels)
eixo_y= np.arange(y_min,y_max,(y_max-y_min)/pixels)
xx,yy = np.meshgrid(eixo_x,eixo_y)
pontos = np.c_[xx.ravel(),yy.ravel()]

z = modelo.predict(pontos)
z= z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.3)
plt.scatter(data_x,data_y,s=1)


