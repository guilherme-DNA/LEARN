import numpy as np
import matplotlib.pyplot as plt

# Gerar dados
red = np.random.randn(50, 2)
blue = np.random.randn(50, 2) + np.array([10, 0])

# Adicionar uma coluna de uns para representar o viés
red = np.column_stack((red, np.ones(50)))
blue = np.column_stack((blue, np.ones(50)))


# Combinar pontos e rótulos
red = np.column_stack((red, np.zeros((50, 1))))
blue = np.column_stack((blue, np.ones((50, 1))))

#learning rate
rate=0.1

#pesos iniciais
pesos=np.random.randn(3)



#valores de entrada
inputs=np.concatenate((red, blue))

#labels dos pontos
labels = inputs[:,3:]


def previsao():
    prev=np.array([])
    for i in range(100):
        prev=np.append(prev,sign(inputs[i,:3],pesos))
    prev=prev.reshape(100,1)
    return prev


def sign(inputs,pesos):
    if np.dot(inputs,pesos) >=0:
        return 1
    return 0

#calculo inicial da previsao
prev=previsao()

#loop de treinamento
while not np.array_equal(prev, labels):
    for i in range(100):
        error = labels[i] - prev[i]
        pesos += rate * error * inputs[i,:3]
    prev=previsao()


print('acabou :)')



red_points = red[:, :2]
blue_points = blue[:, :2]

plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label='Red Points')
plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', label='Blue Points')

# Plotar a linha de decisão
x_values = np.linspace(-3, 8, 100)
y_values = -(pesos[0] / pesos[1]) * x_values - (pesos[2] / pesos[1])  # Ajustado para incluir o termo de viés
plt.plot(x_values, y_values, color='green', linestyle='--', label='Decision Boundary')

plt.legend()
plt.title('Scatter Plot of Red and Blue Points with Decision Boundary')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
