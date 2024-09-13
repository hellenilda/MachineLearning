import numpy as np

def sigmoid(soma):
  return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
  return sig * (1 - sig)

a = sigmoid(0.5)
b = sigmoidDerivada(a) # Gradiente calculado

entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0],[1],[1],[0]])

# Inicializa os pesos entre a camada de entrada e a camada oculta
  # np.random.random((2,3)) cria uma matriz de valores aleatórios entre 0 e 1
    # com 2 neurônios na camada de entrada e 3 neurônios na camada oculta.
  # Multiplicar por 2 e subtrair 1 transforma esses valores para o intervalo [-1, 1],
    # permitindo que os pesos tenham tanto valores positivos quanto negativos.
pesos0 = 2 * np.random.random((2,3)) - 1

# Inicializa os pesos entre a camada oculta e a camada de saída.
  # np.random.random((3,1)) cria uma matriz de valores aleatórios entre 0 e 1
    # com 3 neurônios na camada oculta e 1 neurônio na camada de saída.
  # Multiplicar por 2 e subtrair 1 transforma esses valores para o intervalo [-1, 1],
    # garantindo que os pesos possam assumir valores tanto positivos quanto negativos,
    # o que pode ajudar a rede a convergir mais rapidamente e a explorar melhor o espaço de solução.
pesos0 = 2 * np.random.random((3,1)) - 1


epocas = 10000 # Quantidade de vezes p/ atualizar pesos (erro = 0)
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
  # Cálculos para encontrar a camada oculta
  camadaEntrada = entradas
  somaSinapse0 = np.dot(camadaEntrada, pesos0) # Sinapses são os pesos
  camadaOculta = sigmoid(somaSinapse0)

  # Cálculos para encontrar a saída
  somaSinapse1 = np.dot(camadaOculta, pesos1)
  camadaSaida = sigmoid(somaSinapse1)

  # Cálculo para encontrar o percentual de erro
  erroCamadaSaida = saidas - camadaSaida
    # Percentual de acerto: 1 - mediaAbsoluta
  mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
  print(f'Erro: {mediaAbsoluta}')

  # Cálculo do Delta para a camada de saída
  derivadaSaida = sigmoidDerivada(camadaSaida)
  deltaSaida = erroCamadaSaida * derivadaSaida

  # Cálculo do Delta para a camada oculta
  pesos1Transposta = pesos1.T # A matriz pesos1 passa a ter 3 colunas e 1 linha
  deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
  deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

  # Ajuste dos pesos com backpropagation
  camadaOcultaTransposta = camadaOculta.T
  pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    # peso_n = (peso_n+1 * momento) + (entrada * delta * taxaAprendizagem)
  pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

  camadaEntradaTransposta = camadaEntrada.T
  pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    # peso_n = (peso_n+1 * momento) + (entrada * delta * taxaAprendizagem)
  pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)