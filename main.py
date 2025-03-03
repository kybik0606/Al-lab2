import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r'C:\Users\Admin\PycharmProjects\PythonProject10\processed_titanic10.csv')
print(df.info())
print(df.head())

X = df.drop(columns=['Transported']) # признаки (X)
y = df['Transported'] # целевая переменная (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=5000)
linear_model = LinearRegression()

logistic_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)
y_pred_linear = linear_model.predict(X_test)
y_pred_linear_class = np.where(y_pred_linear >= 0.5, 1, 0)
print("Предсказанные классы (логистическая):", y_pred_logistic[:5])
print("Предсказанные классы (линейная):     ", y_pred_logistic[:5])
print("Истинные классы:                     ", y_test[:5].values)

precision_logistic = precision_score(y_test, y_pred_logistic)
print(f'Precision logistic: {precision_logistic:.2f}')

precision_linear = precision_score(y_test, y_pred_linear_class)
print(f'Precision linear: {precision_linear:.2f}')

print("Отчет классификации (логистическая):")
print(classification_report(y_test, y_pred_logistic))

print("Отчет классификации (линейная):")
print(classification_report(y_test, y_pred_linear_class))

cm_log = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(8,6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок - Логистическая регрессия')

cm_lin = confusion_matrix(y_test, y_pred_linear_class)
plt.figure(figsize=(8,6))
sns.heatmap(cm_lin, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок - Линейная регрессия')
plt.show()