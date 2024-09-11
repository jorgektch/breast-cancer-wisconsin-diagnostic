import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Función para calcular la matriz de confusión
def custom_confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return cm

# Función para calcular métricas
def calculate_metrics(cm):
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Exactitud
    accuracy = (TP + TN) / np.sum(cm)

    # Precisión
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recuperación (Recall)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

# Interfaz de Streamlit
st.title('Clasificación de Cáncer de Mama usando KNN')

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

# Calcular la matriz de confusión
cm = custom_confusion_matrix(y_test, y_pred)

# Calcular las métricas
accuracy, precision, recall, f1 = calculate_metrics(cm)

st.write(f"### Métricas del modelo usando 80% para entrenamiento y 20% para prueba")
st.write(f"- Exactitud (Accuracy): {accuracy:.2f}")
st.write(f"- Precisión (Precision): {precision:.2f}")
st.write(f"- Recuperación (Recall): {recall:.2f}")
st.write(f"- F1-Score: {f1:.2f}")

# Mostrar la matriz de confusión
st.write("#### Matriz de confusión")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Mostrar el código del clasificador KNN
with st.expander("Código del clasificador KNN"):
    st.code('''
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in np.array(X_test)]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    ''', language="python")

# Mostrar el código de la matriz de confusión y métricas
with st.expander("Código para matriz de confusión y métricas"):
    st.code('''
def custom_confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return cm

def calculate_metrics(cm):
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1
    ''', language="python")
