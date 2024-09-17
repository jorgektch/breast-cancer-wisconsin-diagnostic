import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

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

# Función para mostrar métricas de rendimiento
def display_metrics(y_test, y_pred, y_prob, model_name):
    st.write(f"### Desempeño del modelo: {model_name}")
    st.write("Exactitud:", accuracy_score(y_test, y_pred))
    st.write("Precisión:", precision_score(y_test, y_pred, average='macro'))
    st.write("Recuperación (Recall):", recall_score(y_test, y_pred, average='macro'))
    st.write("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    st.write("Matriz de Confusión:")
    st.write(confusion_matrix(y_test, y_pred))

    # Calcular y mostrar AUC
    y_test = y_test.to_numpy()
    auc = roc_auc_score(y_test, y_prob[:, 1], multi_class='ovr')
    st.write("Área Bajo la Curva (AUC):", auc)

    # Graficar la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Curva ROC ({model_name})')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    st.write("\n\n")
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro'), auc

# Interfaz de Streamlit
st.title('Clasificación de Cáncer de Mama usando KNN')

st.write("""
### Descripción del problema:
El objetivo es clasificar los tumores mamarios como malignos o benignos utilizando un modelo K-Nearest Neighbors (KNN) basado en las características de los núcleos celulares.

<div style="text-align: center;">
    <img src="https://cdn.myportfolio.com/bcfbbaaa4fc08b26dd3fcdc1a7bacca6/9002b6a83cabab09a521896e_rw_1200.jpg?h=67a0997f40579778103884702fb3e3d7" alt="Punción Aspiración por Aguja Fina">
    <p>Fuente: Manuel Romera. Cáncer de Mama - Infografías <a href="https://manuelromera.com/cancer-de-mama-infografias">Enlace</a></p>
</div>

<div style="text-align: center;">
    <img src="https://www.redalyc.org/journal/3756/375669596003/375669596003_gf3.png" alt="Muestra de tejido obtenida por PAAF">
    <p>Fuente: Andrés Duque, Ana Karina Ramírez, Jorge Pérez. Punción aspiración con aguja fina guiada por ultrasonido de nódulos mamarios de alta sospecha <a href="https://www.redalyc.org/journal/3756/375669596003/html/">Enlace</a></p>
</div>

### Punción Aspiración por Aguja Fina (PAAF):
La Punción Aspiración por Aguja Fina (PAAF) es un procedimiento diagnóstico utilizado en la evaluación de lesiones mamarias. Consiste en la obtención de una muestra de células del tejido mamario mediante una aguja fina para su posterior análisis bajo el microscopio. La PAAF es mínimamente invasiva y se usa para distinguir entre tumores benignos y malignos.

**Aplicaciones en el diagnóstico de cáncer de mama:**
- **Detección temprana:** La PAAF ayuda en la detección temprana del cáncer de mama, permitiendo el diagnóstico y tratamiento oportunos.
- **Minimización de riesgos:** Ofrece una alternativa menos invasiva en comparación con biopsias más extensas, reduciendo el riesgo para la paciente.
- **Guía para el tratamiento:** La información obtenida a través de PAAF puede ayudar a los médicos a planificar el tratamiento más adecuado basado en el tipo y grado del cáncer.

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
    13. **radius_se**: Desviación estándar del radio.
    14. **texture_se**: Desviación estándar de la textura.
    15. **perimeter_se**: Desviación estándar del perímetro.
    16. **area_se**: Desviación estándar del área.
    17. **smoothness_se**: Desviación estándar de la suavidad.
    18. **compactness_se**: Desviación estándar de la compacidad.
    19. **concavity_se**: Desviación estándar de la concavidad.
    20. **concave_points_se**: Desviación estándar de los puntos cóncavos.
    21. **symmetry_se**: Desviación estándar de la simetría.
    22. **fractal_dimension_se**: Desviación estándar de la dimensión fractal.
    23. **radius_worst**: Peor valor del radio.
    24. **texture_worst**: Peor valor de la textura.
    25. **perimeter_worst**: Peor valor del perímetro.
    26. **area_worst**: Peor valor del área.
    27. **smoothness_worst**: Peor valor de la suavidad.
    28. **compactness_worst**: Peor valor de la compacidad.
    29. **concavity_worst**: Peor valor de la concavidad.
    30. **concave_points_worst**: Peor valor de los puntos cóncavos.
    31. **symmetry_worst**: Peor valor de la simetría.
    32. **fractal_dimension_worst**: Peor valor de la dimensión fractal.

### Tipos de Datos:
- **id**: Entero.
- **diagnosis**: Cadena de texto (M, B).
- **Otros atributos**: Valores reales (float).

### Elección de k=3 en KNN:
El valor de k en KNN define el número de vecinos más cercanos que se consideran para determinar la clase de un punto de datos. La elección de k=3 es común porque proporciona un equilibrio entre suavizar el ruido en los datos y mantener un modelo sensible a la estructura subyacente. Con k=3, el modelo es menos propenso a sobreajustarse a un único vecino ruidoso, pero sigue siendo lo suficientemente sensible como para capturar la variabilidad en los datos.
""", unsafe_allow_html=True)

# Cargar el archivo de datos
df = load_data()
st.write("## Vista previa del dataset")
st.dataframe(df.head())

# Inicializar el modelo KNN y otros objetos en el estado de la sesión
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'column_names' not in st.session_state:
    st.session_state.column_names = []
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None


# Diccionario para almacenar resultados
results = {}
metrics_table = []

# Modelos
models = {
    'Árbol de Decisión': DecisionTreeClassifier(),
    'Red Neuronal Multicapa (MLP)': MLPClassifier(max_iter=500),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Máquina de Vectores de Soporte (SVM - Lineal)': SVC(kernel='linear'),
    'Máquina de Vectores de Soporte (SVM - RBF)': SVC(kernel='rbf'),
    'Random Forest': RandomForestClassifier()
}

X, y, imputer, scaler, pca, column_names = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar y evaluar cada modelo
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else np.zeros((len(X_test), 2))
    
    # Obtener métricas y agregarlas a la tabla
    accuracy, precision, recall, f1, auc = display_metrics(y_test, y_pred, y_prob, model_name)
    metrics_table.append([model_name, accuracy, precision, recall, f1, auc])
    results[model_name] = f1

# Guardar los objetos de preprocesamiento en el estado de la sesión
st.session_state['imputer'] = imputer
st.session_state['scaler'] = scaler
st.session_state['pca'] = pca
st.session_state['column_names'] = column_names

# Comparar los modelos y elegir el mejor
best_model_name = max(results, key=results.get)
st.session_state['best_model_name'] = best_model_name
st.write(f"## El mejor modelo es: {best_model_name} con F1 Score: {results[best_model_name]}")

# Mostrar tabla comparativa de indicadores de performance
st.write("## Comparativa de Indicadores de Desempeño")
df_metrics = pd.DataFrame(metrics_table, columns=['Modelo', 'Exactitud', 'Precisión', 'Recuperación', 'F1 Score', 'AUC'])
st.write(df_metrics)

# Predicción con nuevos datos
st.write("## Predicción con nuevos datos")
new_file = st.file_uploader("Sube un archivo con nuevos datos para predicción (debe tener la misma estructura que el dataset de entrenamiento)", type="csv")

if new_file is not None:
    new_data = pd.read_csv(new_file)

    # Verificar si los objetos ajustados existen
    if imputer is None or scaler is None or pca is None:
        st.error("El modelo no está entrenado. Entrena el modelo primero antes de realizar predicciones.")
    else:
        # Recuperar los objetos del estado de la sesión
        best_model_name = st.session_state['best_model_name']
        imputer = st.session_state['imputer']
        scaler = st.session_state['scaler']
        pca = st.session_state['pca']
        column_names = st.session_state['column_names']

        # Asegurar que las columnas estén en el mismo orden que el entrenamiento
        new_data = new_data.drop(columns=['id'], errors='ignore')  # Eliminar columna ID si está presente
        new_data = new_data.reindex(columns=column_names, fill_value=0)

        # Imputar, normalizar y reducir dimensionalidad de los nuevos datos
        new_data_imputed = imputer.transform(new_data)
        new_data_processed = scaler.transform(new_data_imputed)
        new_data_reduced = pca.transform(new_data_processed)

        # Predecir con el mejor modelo
        best_model = models[best_model_name]
        predictions = best_model.predict(new_data_reduced)

        # Asignar las predicciones al dataset
        new_data['diagnosis'] = np.where(predictions == 1, 'M', 'B')

        st.write("### Predicciones")
        st.dataframe(new_data)
        st.download_button(label="Descargar predicciones", data=new_data.to_csv(index=False), file_name="data_con_prediccion.csv")

# Descargar archivo CSV de ejemplo
example_data = {column: [] for column in column_names}

example_df = pd.DataFrame(example_data)
csv = example_df.to_csv(index=False)

st.download_button(
    label="Descargar archivo CSV de ejemplo",
    data=csv,
    file_name="data_para_prediccion_ejemplo.csv",
    mime="text/csv"
)