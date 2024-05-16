import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Queremos encontrar uma reta boa suficiente para previsão de 1 variável dada outra (regressão com 2 variáveis)

# y = mx + n

# m(t+1) = m(t) - L * derivada parcial em função de m
# n(t+1) = n(t) - L * derivada parcial em função de n

m = 0.15
n = 1
taxa_aprendizado = 0.0001
precisao = 0.001

# Pontos aleatórios próximos da reta
data = np.array([(i, m * i + n + rd.uniform(-4.0, 4.0)) for i in range(100)])

# Plota os pontos
plt.plot(data[:, 0], data[:, 1], 'r*', markersize=4)

# Calcula o erro 1 vez pra se basear no loop
erro_m = np.sum(2 * (m * data[:, 0] + n - data[:, 1]) * data[:, 0]) / len(data)
erro_n = np.sum(2 * (m * data[:, 0] + n - data[:, 1])) / len(data)

print(erro_m, erro_n)

# Aproximação iterativa
while abs(erro_m) > precisao and abs(erro_n) > precisao:
    erro_m = np.sum(2 * (m * data[:, 0] + n - data[:, 1]) * data[:, 0]) / len(data)
    erro_n = np.sum(2 * (m * data[:, 0] + n - data[:, 1])) / len(data)
    m -= taxa_aprendizado * erro_m
    n -= taxa_aprendizado * erro_n

# Gera a reta
x = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)  # 100 pontos de min a max dos dados
y = m * x + n
plt.plot(x, y, label=f'Reta de aproximação')

print(erro_m, erro_n)
plt.legend()
plt.show()
