# model.py

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(features, labels):
    """
    Entrena el modelo de Random Forest con los datos proporcionados.
    """
    X = features.values
    y = labels.values.ravel()
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir el modelo
    rf_model = RandomForestClassifier(random_state=42)
    
    # Definir los hiperparámetros a ajustar
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Búsqueda en cuadrícula
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    print('Mejores hiperparámetros:', grid_search.best_params_)
    
    return best_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y muestra métricas de rendimiento.
    """
    y_pred = model.predict(X_test)
    
    # Exactitud
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Exactitud del modelo: {accuracy * 100:.2f}%')
    
    # Reporte de clasificación
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print('Matriz de confusión:')
    print(cm)
    
    # Visualizar matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Actual')
    plt.show()

def save_model(model, filename):
    """
    Guarda el modelo entrenado en un archivo.
    """
    joblib.dump(model, filename)
    print(f'Modelo guardado en {filename}')
