import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Mostrar todas as colunas do DataSet CSV
pd.set_option('display.max_columns', 21)

# Importando DataSet
dataSet = pd.read_csv('C:/curso.de.machine.learning/src/LinearRegression/Helpers/DataSets/price_house_data/kc_house_data.csv')

# Exluir features irrelevantes
dataSet.drop('id', axis=1, inplace=True)
dataSet.drop('date', axis=1, inplace=True)
dataSet.drop('zipcode', axis=1, inplace=True)
dataSet.drop('lat', axis=1, inplace=True)
dataSet.drop('long', axis=1, inplace=True)

# Definir as variaveis preditoras e variavel target
# Variaveis preditoras
x = dataSet.drop('price', axis=1)
# Variaveis target
y = dataSet['price']

# Divisão de dados entre treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Criando o modelo de treino Regressão linear
model = linear_model.LinearRegression()
# Aplicando dados de treino ao modelo para encontrar o valor de a e b (y = a + b.x)
model.fit(x_train, y_train)

# Capturando o valor do coeficiente angular (a)
# print('(a) Interceptos coeficiente angular: ', model.intercept_)

# Capturando o valor do coeficiente linear (b)
# print('(b) Inclinação coeficiente linear: ', model.coef_)

# Exibindo reta de regresão no DataSet de treino
plt.scatter(x_train['bedrooms'], y_train)

plt.plot(x_train, model.coef_*x_train - model.intercept_, '-r')
plt.xlabel('Casa Treino')
plt.ylabel('Preço Treino')
plt.show()

# --------------------Executando o modelo no DataSet de teste--------------------

# Calculando as predições usando o modelo e base de teste
predictPriceTest = model.predict(x_test)

# Exibindo reta de regresão no DataSet de teste
plt.scatter(x_test['bedrooms'], y_test)

plt.plot(x_test, model.coef_*x_test - model.intercept_, '-r')
plt.xlabel('Casa Teste')
plt.ylabel('Preço Teste')
plt.show()


# --------------------Avaliando o Modelo--------------------

# Calculando o coeficiente de determinação R2 com dados de teste
# resultR2 = model.score(x_test, y_test)
print('Soma dos Erros ao Quadrado (SSE): ', np.sum((predictPriceTest - y_test)**2))
print('Erro Quadrático Médio (MSE): ', mean_squared_error(y_test, predictPriceTest))
print("Erro Médio Absoluto (MAE): ", mean_absolute_error(y_test, predictPriceTest))
print('Raiz do Erro Quadrático Médio (RMSE): ', sqrt(mean_squared_error(y_test, predictPriceTest)))
print('R2-Score: ', r2_score(predictPriceTest, y_test))





