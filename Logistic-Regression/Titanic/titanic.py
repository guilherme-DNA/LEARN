import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Carregar o arquivo CSV em um DataFrame pandas
df = pd.read_csv('train.csv')
media_idade = df['Age'].mean()

# Preencher os valores ausentes na coluna 'Age' com a média da idade
df['Age'] = df['Age'].fillna(media_idade)
# Converter o DataFrame em um array numpy
data = df.to_numpy()
data = data[:,1:] # Removendo o ID do passageiro (pq nn tem nenhuma relação pra se sobreviveu ou nn)

#queremos classificar dados de entrada em 2 grupos, resultado discreto (sim/nao , vive/morre , etc)

#sigmoide  = 1/ 1 + e^(b0 + b1x)   , x=[0,1]

#funcao de custo (aproximador dos pesos) = cross-entropy loss

variaveis=['Pclass','Sex','Age','SibSp','Parch','Fare']

ref_data={'survive':0,'Pclass':1,'Sex':3 , 'Age':4 , 'SibSp':5 , 'Parch':6 , 'Fare':8 }  #referencial pra cada Variavel


mapeamento = {'male': 0, 'female': 1}
data[:, ref_data['Sex']] = np.array([mapeamento[val] for val in data[:, ref_data['Sex']]]) #trocando male e female pra [0,1]

beta0=2  #intercepto (valor base q entra no calculo de z)

pesos=np.ones(9) #pesos a serem estimados

N=len(data[:,0])

taxa_aprendizado = 0.002    #taxa de passo dos pesos

precisao=0.1           #valor minimo de um passo para encerrar o treinamento


def sigmoide(z):
    return 1 / (1 + np.exp(-z))

#print([data[i, ref_data['Pclass']] for i in range(5)])


def error(pesos):
    res = np.zeros_like(pesos)
    for i in range(N):
        z = beta0
        for var in variaveis:
            z += pesos[ref_data[var]] * data[i, ref_data[var]]
        y_chapeu = sigmoide(z)
        erro = -1 / N * (data[i, ref_data['survive']] - y_chapeu)
        for var in variaveis:
            res[ref_data[var]] += erro * data[i, ref_data[var]]
    return res


inicio = time.time()


#calculamos o primeiro erro de tds variaveis
gradiente=error(pesos)
pesos -= taxa_aprendizado * gradiente


#Loop de treinamento
while np.sum(np.abs(gradiente))>precisao:
    gradiente=error(pesos)
    pesos -= taxa_aprendizado * gradiente
    



fim = time.time()

# Calcula o tempo de execução
tempo_execucao = fim - inicio



#-----------------------------------------------------------------------plotando --------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(data[:, ref_data['Age']], data[:, ref_data['survive']], color='blue', label='Data')
plt.xlabel('Idade')
plt.ylabel('Sobrevivência')
x_values = np.linspace(min(data[:, ref_data['Age']]), max(data[:, ref_data['Age']]), 100)
z_values = beta0 + pesos[ref_data['Age']] * x_values
y_values = sigmoide(z_values)
plt.plot(x_values, y_values, color='red', label='Sigmoide')
plt.legend()
plt.show()


#------------------------------------------------------testando dados apos treinamento --------------------------------------------------------------------


df_novos_dados = pd.read_csv('test.csv')
media_idade = df_novos_dados['Age'].mean()
df_novos_dados['Age'] = df_novos_dados['Age'].fillna(media_idade)
mapeamento = {'male': 0, 'female': 1}
df_novos_dados['Sex'] = df_novos_dados['Sex'].map(mapeamento)


def prever_sobrevivencia(dados, indice):
    z = beta0
    for var in variaveis:
        z += pesos[ref_data[var]] * dados[var][indice]
    y_hat = sigmoide(z)
    if y_hat > 0.5:
        return 1
    else:
        return 0


previsoes = []


# Loop para prever dados de teste
for i in range(len(df_novos_dados)):
    sobreviveu = prever_sobrevivencia(df_novos_dados, i)
    previsoes.append(sobreviveu)


# Adicionar as previsões ao DataFrame
df_novos_dados['Survived'] = previsoes
# Criar DataFrame com apenas as colunas de ID do passageiro e sobrevivência prevista
df_resultado = df_novos_dados[['PassengerId', 'Survived']]
# Salvar os resultados em um novo arquivo CSV
df_resultado.to_csv('resultado_previsto3.csv', index=False)
print("Tempo de execução:", tempo_execucao, "segundos")
