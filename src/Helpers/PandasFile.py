import pandas as pd

data = pd.read_excel('C:/curso.de.machine.learning/src/file/excel_example.xlsx')

# Obter cabeçalhos
print(data.head(11))