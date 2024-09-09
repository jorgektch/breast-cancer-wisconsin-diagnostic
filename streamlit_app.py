import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
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
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
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

# Preprocesamiento de datos
def preprocess_data(df):
    df = df.drop(columns=['id'])  # Eliminar columna ID
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convertir la columna diagnosis a valores binarios
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    # Normalización de los datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Interfaz de Streamlit
st.title('Clasificación de Cáncer de Mama usando KNN')

st.write("""
### Descripción del problema:
El objetivo es clasificar los tumores mamarios como malignos o benignos utilizando un modelo K-Nearest Neighbors (KNN) basado en las características de los núcleos celulares.

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
""")

# Cargar el archivo de datos
df = load_data()
st.write("## Vista previa del dataset")
st.dataframe(df.head())

# Preprocesamiento de datos
X, y = preprocess_data(df)

# Dividir los datos en entrenamiento y prueba (80% y 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("""
### Implementación del algoritmo KNN
El siguiente código implementa el algoritmo KNN desde cero utilizando la distancia euclidiana:
""")
st.code('''
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
''', language='python')

# Entrenamiento del modelo
knn = KNN(k=3)

if st.button("Entrenar modelo"):
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"### Métricas del modelo")
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
    new_data_processed = scaler.transform(new_data.drop(columns=['id']))  # Preprocesar
    predictions = knn.predict(new_data_processed)

    # Asignar las predicciones al dataset
    new_data['diagnosis'] = np.where(predictions == 1, 'M', 'B')

    st.write("### Predicciones")
    st.dataframe(new_data)
    st.download_button(label="Descargar predicciones", data=new_data.to_csv(index=False), file_name="predicciones.csv")
