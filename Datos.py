import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de gastos ''')
st.image("Imagen.jpeg", caption="El Titanic navegaba desde Southampton, Inglaterra, hasta Nueva York en Estados Unidos.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  Presupuesto = st.number_input('Presupuesto:', min_value=0.0, max_value=1000000.0, value = 0.0, step = 1.0)
  Tiempo_invertido = st.number_input('Tiempo invertido:', min_value=0.0, max_value=100000.0, value = 0.0, step = 1.0)
  Tipo = st.number_input('Tipo (Alimentos/salud, ahorro/inversión, ejercicio/deporte, entretenimiento/ocio, académico, transporte):', min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)
  Momento = st.number_input('Momento',min_value=0.0, max_value=3.0, value = 0.0, step = 1.0)
  No. de personas = st.number_input('No. de personas:', min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)

  user_input_data = {'Presupuesto': Presupuesto,
                     'Tiempo invertido': Tiempo_invertido,
                     'Tipo (Alimentos/salud, ahorro/inversión, ejercicio/deporte, entretenimiento/ocio, académico, transporte)': Tipo,
                     'Momento': Momento,
                     'No. de personas': No. de personas}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('Datos.csv', encoding='latin-1')
X = datos.drop(columns='Costo')
y = datos['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613555)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo (Alimentos/salud, ahorro/inversión, ejercicio/deporte, entretenimiento/ocio, académico, transporte)'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas'] 

st.subheader('Calculo de gasto')
st.write('El gasto será: ', prediccion)
