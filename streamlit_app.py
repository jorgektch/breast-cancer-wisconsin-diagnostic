import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Función para calcular la distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Implementación del clasificador KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)  # Convertir a array para asegurar consistencia
        self.y_train = np.array(y_train)  # Convertir a array para asegurar consistencia
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in np.array(X_test)]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Cargar datos desde el archivo data.csv en la misma carpeta
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    return df

# Preprocesamiento de datos mejorado
def preprocess_data(df):
    df = df.drop(columns=['id'])  # Eliminar columna ID
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convertir la columna diagnosis a valores binarios

    # Eliminación de características menos relevantes
    df = df.drop(columns=['fractal_dimension_mean', 'fractal_dimension_worst', 'fractal_dimension_se'])

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis'] if 'diagnosis' in df.columns else None

    # Imputación de valores faltantes con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normalización de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced, y, imputer, scaler, pca, X.columns  # Devolver los nombres de las columnas

# Interfaz de Streamlit
st.title('Clasificación de Cáncer de Mama usando KNN')

st.write("""
### Descripción del problema:
El objetivo es clasificar los tumores mamarios como malignos o benignos utilizando un modelo K-Nearest Neighbors (KNN) basado en las características de los núcleos celulares.

### Punción Aspiración por Aguja Fina (PAAF):
La Punción Aspiración por Aguja Fina (PAAF) es un procedimiento diagnóstico utilizado en la evaluación de lesiones mamarias. Consiste en la obtención de una muestra de células del tejido mamario mediante una aguja fina para su posterior análisis bajo el microscopio. La PAAF es mínimamente invasiva y se usa para distinguir entre tumores benignos y malignos.

**Aplicaciones en el diagnóstico de cáncer de mama:**
- **Detección temprana:** La PAAF ayuda en la detección temprana del cáncer de mama, permitiendo el diagnóstico y tratamiento oportunos.
- **Minimización de riesgos:** Ofrece una alternativa menos invasiva en comparación con biopsias más extensas, reduciendo el riesgo para la paciente.
- **Guía para el tratamiento:** La información obtenida a través de PAAF puede ayudar a los médicos a planificar el tratamiento más adecuado basado en el tipo y grado del cáncer.

**Imágenes de PAAF:**

![Punción Aspiración por Aguja Fina](https://upload.wikimedia.org/wikipedia/commons/7/72/Needle_biopsy.png)
*Fuente: [Wikipedia](https://en.wikipedia.org/wiki/File:Needle_biopsy.png)*

![Muestra de tejido obtenida por PAAF](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Breast_needle_biopsy_02.jpg/1280px-Breast_needle_biopsy_02.jpg)
*Fuente: [Wikipedia](https://en.wikipedia.org/wiki/File:Breast_needle_biopsy_02.jpg)*

### Dataset:
Este dataset describe las características de los núcleos celulares presentes en una imagen obtenida a partir de una aspiración con aguja fina (AAF) de una masa mamaria.

- **Fuente Kaggle**: [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?select=data.csv)
- **Fuente UC Irvine Machine Learning Repository**: [Wisconsin Breast Cancer Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Descripción del Dataset:
- **Número de registros**: 569
- **Nombres de las columnas y su descripción**:
    1. **id**: Número de identificación (no utilizado en el modelo).
    2. **diagnosis**: Diagnóstico (M = maligno, B = benigno).
    3. **radius_mean**: Media de distancias desde el centro hasta los puntos en el perímetro.
    4. **texture_mean**: Desviación estándar de los valores de escala de grises.
    5. **perimeter_mean**: Perímetro.
    6. **area_mean**: Área.
    7. **smoothness_mean**: Variación local en la longitud del radio.
    8. **compactness_mean**: (perímetro^2 / área - 1.0).
    9. **concavity_mean**: Severidad de las porciones cóncavas del contorno.
    10. **concave_points_mean**: Número de porciones cóncavas del contorno.
    11. **symmetry_mean**: Simetría.
    12. **fractal_dimension_mean**: "Aproximación de la línea de costa" - 1.
    13. ...: (y otros atributos similares basados en el análisis de imágenes).

### Tipos de Datos:
- **id**: Entero.
- **diagnosis**: Cadena de texto (M, B).
- **Otros atributos**: Valores reales (float).

### Elección de k=3 en KNN:
El valor de k en KNN define el número de vecinos más cercanos que se consideran para determinar la clase de un punto de datos. La elección de k=3 es común porque proporciona un equilibrio entre suavizar el ruido en los datos y mantener un modelo sensible a la estructura subyacente. Con k=3, el modelo es menos propenso a sobreajustarse a un único vecino ruidoso, pero sigue siendo lo suficientemente sensible como para capturar la variabilidad en los datos.
""")

# Cargar el archivo de datos
df = load_data()
st.write("## Vista previa del dataset")
st.dataframe(df.head())

# Inicializar el modelo KNN y otros objetos en el estado de la sesión
if 'knn' not in st.session_state:
    st.session_state.knn = KNN(k=3)
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'column_names' not in st.session_state:
    st.session_state.column_names = []

# Entrenamiento del modelo
X, y, imputer, scaler, pca, column_names = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar los objetos ajustados en el estado de la sesión
st.session_state.knn.fit(X_train, y_train)
st.session_state.imputer = imputer
st.session_state.scaler = scaler
st.session_state.pca = pca
st.session_state.column_names = list(column_names)  # Guardar nombres de columnas para validación

# Evaluar el modelo
y_pred = st.session_state.knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write(f"### Métricas del modelo usando 80% para entrenamiento y 20% para prueba")
st.write(f"- Precisión (Accuracy): {accuracy:.2f}")
st.write(f"- F1-Score: {f1:.2f}")

st.write("#### Matriz de confusión")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.write("#### Reporte de Clasificación")
report = classification_report(y_test, y_pred, target_names=["Benigno", "Maligno"])
st.text(report)

# Predicción con nuevos datos
st.write("## Predicción con nuevos datos")
new_file = st.file_uploader("Sube un archivo con nuevos datos para predicción (debe tener la misma estructura que el dataset de entrenamiento)", type="csv")

if new_file is not None:
    new_data = pd.read_csv(new_file)

    # Verificar si los objetos ajustados existen en el estado de la sesión
    if st.session_state.imputer is None or st.session_state.scaler is None or st.session_state.pca is None:
        st.error("El modelo no está entrenado. Entrena el modelo primero antes de realizar predicciones.")
    else:
        # Asegurar que las columnas estén en el mismo orden que el entrenamiento
        new_data = new_data.drop(columns=['id'], errors='ignore')  # Eliminar columna ID si está presente
        new_data = new_data.drop(columns=['diagnosis'], errors='ignore')  # Eliminar columna diagnosis si está presente
        new_data = new_data.reindex(columns=st.session_state.column_names, fill_value=0)

        # Imputar, normalizar y reducir dimensionalidad de los nuevos datos
        new_data_imputed = st.session_state.imputer.transform(new_data)
        new_data_processed = st.session_state.scaler.transform(new_data_imputed)
        new_data_reduced = st.session_state.pca.transform(new_data_processed)

        # Predecir con el modelo KNN
        predictions = st.session_state.knn.predict(new_data_reduced)

        # Asignar las predicciones al dataset
        new_data['diagnosis'] = np.where(predictions == 1, 'M', 'B')

        st.write("### Predicciones")
        st.dataframe(new_data)
        st.download_button(label="Descargar predicciones", data=new_data.to_csv(index=False), file_name="predicciones.csv")
