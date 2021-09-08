from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Função make_regression para criar conjuntos de dados
x, y = make_regression(n_samples=200, n_features=1, noise=30)
print(y)

# Funcão de divisão de dados entre treino e teste com coeficiente de determinação R2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Criando o modelo Regressão linear
model = LinearRegression()

# Aplicando dados de treino ao modelo
model.fit(x_train, y_train)

# Função que retorna o coeficiente de determinação R2 com dados de teste
result = model.score(x_test, y_test)
print(result)

# Função y = mx+b Capturando (m = coeficiente angular)
model.coef_

# Capturando (b = coeficiente linear)
model.intercept_

# Função y = mx+b Mostrado os dados da Regressão linear
plt.scatter(x_train, y_train)
xreg = np.arange(-3, 3, 1)

# Plotando Dados de treino
plt.plot(xreg, model.coef_[0]*xreg - model.intercept_, color='red')

# plt.show()
