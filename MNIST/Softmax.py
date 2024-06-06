import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


mnist = fetch_openml( 'mnist_784' ) 
mnist_data = mnist.data.to_numpy()
mnist_target = mnist.target.to_numpy()




# imagens de treino e teste
train_data = mnist_data[:60000,:]         #train_data.shape() = (60000,784)
test_data= mnist_data[60000: , : ]        #train_data.shape() = (10000,784)


# labels de treino e teste
train_labels = mnist_target[:60000].astype(int)       #train_labels.shape() = (60000,)
test_labels = mnist_target[60000:].astype(int)        #train_labels.shape() = (10000,)




N=train_data.shape[0]

pesos = np.random.rand(10, 784)             #(10,784)


taxa_aprendizado = 0.0002    #taxa de passo dos pesos


#print(train_data[2].shape)


def softmax(imagem):
    b = np.dot(pesos, train_data[imagem])             # (10, 784) * (784,) -> (10,)
    b -= np.max(b)  # Estabilização numérica
    a = np.exp(b)
    return a / np.sum(a)



def error():
    prevs = np.zeros((N, 10))
    for i in range(N):                    #loop para gera matriz com todas probs de tds imagens  ---->  necessaria para one hot encoding otimizado
        prevs[i] = softmax(i)           

    
    one_hot_labels = np.zeros(prevs.shape)
    for i in range(train_labels.size):              #one hot encoding 
        one_hot_labels[i, train_labels[i]] = 1

        
    error = prevs - one_hot_labels                  
    gradients = np.dot(error.T, train_data) / train_data.shape[0]            #1/N
    return gradients


def predict(data):
    scores = np.dot(pesos, data.T)                  #gera as probs de tds imagens
    return np.argmax(scores, axis=0)                #pega a maior prob de cada imagem e gera um array (60000,) para comparacao com labels


#-----------------loop de otimizacao com N épocas------------------------


epoch=1000      #numero de épocas

values=[]

n=list(range(epoch))


for e in range(epoch):
    gradients = error()
    pesos -= taxa_aprendizado * gradients
    print(e)
    predicoes = predict(test_data)
    precisao = np.mean(predicoes == test_labels)
    values.append(precisao)
    print(f'o treinamento resultou em uma precisão de {precisao}')


#-------------------------teste após treino (com print da precisão final) -----------------------------

plt.plot(n,values)
plt.xlabel('épocas')
plt.ylabel('precisão (acertos/total)')
plt.title('Progresso do modelo ao longo do treinamento')
plt.show()

predicoes = predict(test_data)
precisao = np.mean(predicoes == test_labels)    #retorna um valor [0,1] do percentual de imagens com label prevista corretamente

print(f'o treinamento resultou em uma precisão FINAL de {precisao}')
