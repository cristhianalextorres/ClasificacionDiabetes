# -*- coding: utf-8 -*-


# Carga de Libreria
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import log_loss

"""
# Introducción a la equidad

***Cristhian Alexander Torres Polanco***

**Revisión de Sesgos**

En el presente contenido revisaremos si el dataset prueba de siguiente ejercicio posee sesgos que favorescan o discriminen a una población.

**Conjuto de datos sobre Diabetes**

El conjunto de datos objeto de prueba se extrae de la pagina **Kaggle**, en cual es una recopilación de datos que proviene del instituto nacional de Diabetes y Enfermedades Digestivas y Renales de la India. La fecha de recepción de los datos es **9 de mayo de 1990** y es compone 768 observaciones, 8 caracteristicas y una clase.

Enlace del data set: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

**Descripcion del data set**

El contenido del conjuto de datos se refiere a observaciones de pacientes, todas mujeres, entre los 21 a 81 años de edad de ascedencia india Pima.

**Caracteristicas:**
* Pregnancies: Número de veces que está embarazada
* Glucose: Concentración de glucosa plasmática a las 2 horas en una prueba de tolerancia oral a la glucosa
* BloodPressure: presión arterial diastólica (mm Hg)
* SkinThickness: Grosor del pliegue cutáneo del tríceps (mm)
* Insulin: insulina sérica de 2 horas (mu U/ml)
* BMI: índice de masa corporal (peso en kg/(altura en m)^2)
* DiabetesPedigreeFunction: Función de pedigrí de diabetes
* Age: Edad (años)

**Target u Objetivo**

El target se representa por el campo **Outcome** que contiene valores de 1 y 0; los cuales describen 1 para positivo (paciente con diabetes) y 0 negativo (paciente sin diabetis)

"""
# Cargar datos
df = pd.read_csv('diabetes.csv')
st.write(df.head())

"""**Entrenamiento del Modelo:** Se entrena el modelo de regresión logistíca con parametros por Defoult. Para ello, se escala el data set con el objetivo de mejorar en desempeño de la regresión."""

#Separación de la data train y test

X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#Se escala el dataset:
scaler = StandardScaler()
#dfScaler = pd.DataFrame(scaler)

xTrainScaler = scaler.fit_transform(X_train)
xTestScaler = scaler.fit_transform(X_test)

dfScaler = pd.DataFrame(xTrainScaler)
st.write(dfScaler.head())

# Se carga el modelo Regresión Logística
logistica = LogisticRegression(max_iter=50, warm_start=True, solver='lbfgs')
# Se entrena el modelo
logistica.fit(xTrainScaler, y_train)
# Se obtiene el accurancy
acc = str(logistica.score(xTestScaler, y_test))

st.markdown(
"""**Exactitud del Modelo:**  El modelo sin hiperparametros ajustados efectua el **"""+acc+"""** de exactitud. El desempeño no es el mejor."""
)


# Listas para almacenar el costo en cada iteración
train_cost_history = []
test_cost_history = []

# Entrenar el modelo y calcular la función de costo en cada iteración
for i in range(50):  # Número de iteraciones
    logistica = LogisticRegression(max_iter=i, warm_start=True, solver='lbfgs')
    logistica.fit(xTrainScaler, y_train)
    
    # Calcular y registrar la función de costo en el conjunto de entrenamiento
    y_train_pred_prob = logistica.predict_proba(xTrainScaler)
    train_cost = log_loss(y_train, y_train_pred_prob)
    train_cost_history.append(train_cost)
    
    # Calcular y registrar la función de costo en el conjunto de prueba
    y_test_pred_prob = logistica.predict_proba(xTestScaler)
    test_cost = log_loss(y_test, y_test_pred_prob)
    test_cost_history.append(test_cost)

# Crear la figura de Matplotlib
figura, ax = plt.subplots()
ax.plot(range(1, len(train_cost_history) + 1), train_cost_history, marker='x', label='Train Cost')
ax.plot(range(1, len(test_cost_history) + 1), test_cost_history, marker='x', label='Test Cost')
ax.set_title('Función de Costo Regresión Logística')
ax.set_xlabel('Iteración')
ax.set_ylabel('Log-Loss')
ax.grid(True)
ax.legend()

ax.grid(True)

# Mostrar la gráfica en Streamlit
st.pyplot(figura)



fig, a = plt.subplots(figsize=(6, 3))
vReport = ClassificationReport(logistica)
vReport.fit(xTrainScaler, y_train)
vReport.score(xTestScaler, y_test)
fig = vReport.fig
st.pyplot(fig)

figura, a = plt.subplots(figsize=(6, 6))
vROC2 = ROCAUC(logistica)
vROC2.fit(xTrainScaler, y_train)
vROC2.score(xTestScaler, y_test)
figura = vROC2.fig
st.pyplot(figura)

st.markdown(
"""**Matriz de confusión:** la matriz hace una comparación entre los valores predecidos y los valores reales. Se divide de la siguiente forma:

Se debe tener presente que en el ejercicio. Las personas con diagnostico 0 son negativas para diabetes y con 1 para las personas que tienen diabetes.

* Verdaderos Positivos: Personas que son negativas para diabetes y el modelo las clasificó como negativas. (Siguente gráfico: Superior Izquierdo)
* Verdaderon negativo: Personas que son positivas para diabetes y el modelo las clasificó como positivas. (Siguente gráfico: Inferior derecho)
* Falso negativo: Personas que son negativo para diabetes y el modelo las clasificó como positivas. (Siguente gráfico: Inferior izquierdo)
* Falso Positivo: Personas que son positivas para diabetes y el modelo las clasificó como Negativo. (Siguente gráfico: Superior Derecho)

En este caso se observa que el modelo tiene mayor efectividad para clasificar los Negativos que los positivos.
""")

f, a = plt.subplots(figsize=(6, 6))
# Cargar y evaluar la matriz de confusión de acuerdo al modelo y data de prueba.
metrics.ConfusionMatrixDisplay.from_estimator(logistica, xTestScaler, y_test, cmap=plt.cm.Blues, ax=a)
st.pyplot(f)


