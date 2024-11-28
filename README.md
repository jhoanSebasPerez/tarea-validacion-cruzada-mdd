# **Predicción de Emisiones de CO₂ en Vehículos Pesados**

Este proyecto utiliza **Streamlit** para crear una aplicación interactiva que predice las emisiones de CO₂ en vehículos pesados. Se implementan varios modelos de Machine Learning para realizar predicciones, validaciones cruzadas y visualizaciones.

## **Requisitos Previos**

1. **Python instalado** (versión 3.7 o superior). Descárgalo desde [python.org](https://www.python.org/).
2. Conexión a internet para instalar dependencias.

---

## **Instalación**

1. **Clona o descarga el repositorio:**
   ```bash
   git clone https://github.com/tu-repositorio/prediccion-co2.git
   cd prediccion-co2
   ```

2. **Crea un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv streamlit_env
   ```
   Activa el entorno:
   - En **Windows**:
     ```bash
     streamlit_env\\Scripts\\activate
     ```
   - En **macOS/Linux**:
     ```bash
     source streamlit_env/bin/activate
     ```

3. **Instala las dependencias necesarias:**
   ```bash
   pip install -r requirements.txt
   ```

   **Nota:** Asegúrate de tener un archivo `requirements.txt` con el siguiente contenido:
   ```
   streamlit
   pandas
   matplotlib
   scikit-learn
   numpy
   ```

4. **Asegúrate de que el archivo `vehiculos_procesado.csv` esté en el directorio principal del proyecto.**

---

## **Ejecución de la Aplicación**

1. **Ejecuta la aplicación Streamlit:**
   ```bash
   streamlit run app.py
   ```

2. **Carga del archivo CSV:**
   - Al iniciar la aplicación, se abrirá una interfaz en tu navegador.
   - Sube el archivo `vehiculos_procesado.csv` desde la interfaz.

3. **Explora las funciones de la aplicación:**
   - **Matriz de Correlación:** Visualiza la relación entre las variables del dataset.
   - **Predicciones:** Compara los resultados de los modelos:
     - Naive Bayes.
     - Random Forest.
     - Regresión Lineal.
     - KNN.
   - **Validación Cruzada:** Evalúa los modelos utilizando R² y validación cruzada con 10 repeticiones.
   - **Optimización de Hiperparámetros:** Ajusta los parámetros de Random Forest.
   - **Gráficos:** Visualiza la dispersión de los valores reales vs predichos.

---

## **Estructura del Proyecto**

```
prediccion-co2/
│
├── app.py                  # Script principal de la aplicación
├── vehiculos_procesado.csv # Dataset con las variables: consumo, co2, cilindros, desplazamiento
├── requirements.txt        # Lista de dependencias necesarias
├── README.md               # Instrucciones del proyecto
```

---

## **Problemas Comunes**

1. **El archivo `vehiculos_procesado.csv` no se encuentra:**
   - Asegúrate de que el archivo esté en el mismo directorio que `validacion-cruzada.py`.

2. **Error al instalar dependencias:**
   - Verifica que estás usando la versión correcta de Python (3.7 o superior).
   - Actualiza `pip`:
     ```bash
     pip install --upgrade pip
     ```

3. **Streamlit no abre en el navegador:**
   - Abre la URL manualmente desde la terminal (ejemplo: `http://localhost:8501`).

---

## **Personalización**

1. Cambia los hiperparámetros de los modelos en el código (archivo `app.py`).
2. Agrega nuevos gráficos o funciones para analizar los resultados.

---

## **Licencia**

Este proyecto está bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.
"""