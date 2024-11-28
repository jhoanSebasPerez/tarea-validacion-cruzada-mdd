import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Título de la aplicación
st.title("Predicción de Emisiones de CO₂ en Vehículos Pesados")

# Paso 1: Subir archivo CSV
current_directory = os.getcwd()
file_name = "vehiculos_procesado.csv"  # Nombre del archivo esperado
file_path = os.path.join(current_directory, file_name)

st.write(f"### Cargando datos del archivo : {file_name}")
if os.path.isfile(file_path):
    # Leer los datos
    st.write("### Datos cargados:")
    data = pd.read_csv(file_path, usecols=["consumo", "co2", "cilindros", "desplazamiento"])
    st.write(data.head())

    # Paso 2: Matriz de correlación
    st.header("2. Matriz de Correlación")
    st.write("### Matriz de correlación:")
    st.write(data.corr())

    # Paso 3: División de datos
    st.header("3. División de datos en Entrenamiento y Prueba")
    X = data[["consumo", "cilindros", "desplazamiento"]]
    y = data["co2"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    st.write(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    st.write(f"Datos de prueba: {X_test.shape[0]} muestras")

    # Convertir 'y' en categorías para Naive Bayes
    y_categorico = pd.cut(y, bins=3, labels=["Bajo", "Medio", "Alto"])

    # Paso 4: Modelos y Predicciones
    st.header("4. Predicción con Modelos")
    models = {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Regresión Lineal": LinearRegression(),
        "KNN": KNeighborsRegressor()
    }

    predictions = {}
    for name, model in models.items():
        if "Naive Bayes" in name:
            # Entrenar GaussianNB con etiquetas categorizadas
            model.fit(X_train, y_categorico.loc[y_train.index])  # Usar índices de y_train
            predictions[name] = model.predict(X_test)  # Predice categorías
        else:
            # Entrenar otros modelos con la variable continua
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_test)

        st.write(f"Modelo {name} entrenado.")
    
    # Mostrar advertencia para Naive Bayes
    st.write("Nota: Naive Bayes predice categorías ('Bajo', 'Medio', 'Alto') en lugar de valores continuos.")
        

    # Paso 5: Validación Cruzada
    st.header("5. Validación Cruzada (10 repeticiones)")

    # Validación para GaussianNB (clasificación)
    st.subheader("Validación para Naive Bayes")
    scores_nb = cross_val_score(GaussianNB(), X_scaled, y_categorico, cv=10, scoring='accuracy')
    st.write(f"Naive Bayes (clasificación) - Promedio Accuracy: {scores_nb.mean():.4f}")

    # Validación para modelos de regresión
    st.subheader("Validación para Modelos de Regresión")
    for name, model in models.items():
        if "Naive Bayes" not in name:
            try:
                scores = cross_val_score(model, X_scaled, y, cv=10, scoring='r2')
                st.write(f"{name} - Promedio R²: {scores.mean():.4f}")
            except Exception as e:
                st.write(f"{name} - Error durante la validación cruzada: {str(e)}")

    # Paso 6: Optimización de Hiperparámetros
    st.header("6. Optimización de Hiperparámetros para Random Forest")
    param_grid_rf = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=10, scoring='r2')
    grid_rf.fit(X_train, y_train)
    st.write("Mejores parámetros para Random Forest:")
    st.write(grid_rf.best_params_)
    st.write(f"Mejor R²: {grid_rf.best_score_:.4f}")

    # Paso 7: Gráficos de dispersión
    st.header("7. Dispersión CO₂ Actual vs Predicción")
    for name, y_pred in predictions.items():
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
        ax.set_xlabel("CO₂ Real")
        ax.set_ylabel("CO₂ Predicción")
        ax.set_title(f"CO₂ Actual vs Predicción - {name}")
        st.pyplot(fig)