import pandas as pd 

SEED = 20
uri = "./tracking.csv"  # importing a dataset into an variable
dados = pd.read_csv(uri, sep=",")   # usando pandas para ler o data set
print(dados.head())  # print das primeiras linhas


# escolhendo certas colunas e separando em variaveis
x = dados[["home","how_it_works", "contact"]]
y = dados[["bought"]]

# renomeando colunas a gosto
mapa = {
    "home" :"principal",
    "how_it_works" :"sobre",
    "contact" : "contato",
    "bought":"comprou"
}
dados = dados.rename(columns = mapa)
print(dados.head())
x = dados[["principal","sobre", "contato"]]
y = dados[["comprou"]]

# separando treino de teste:
from sklearn.model_selection import train_test_split

print(dados.shape) # vendo qual o tamanho dos nossos dados
# usando a biblioteca par separar treino e teste
treino_x, teste_x, treino_y , teste_y = train_test_split(x,y, random_state = SEED, test_size = 0.25, stratify = y)
print(teste_y.value_counts())
print(treino_y.value_counts()) #tem que ter cuidado com essa proporção de quantos compram p ele n aprender errado
# tem que aprender com a mesma proporçã que tem no treino aí usamos o stratify


# hora de treinar:
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


model = LinearSVC()
model.fit(treino_x,treino_y)
previsoes = model.predict(teste_x)

# Comparando as previsoes
acuracia = accuracy_score(teste_y,previsoes)
print("taxa de acerto %.2f" % acuracia)
print(previsoes)



from sklearn.model_selection import train_test_split

treino_x, teste_x, treino_y , teste_y = train_test_split(x,y,test_size = 0.25)
