# model_evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo entrenado y muestra resultados en porcentaje.

    Args:
        model: Modelo entrenado.
        X_test (array): Características de prueba.
        y_test (array): Etiquetas de prueba.

    Returns:
        dict: Resultados de la evaluación.
    """
    # Obtener las probabilidades de predicción
    y_pred_proba = model.predict(X_test)
    
    # Convertir las probabilidades en etiquetas de clase
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calcular precisión en porcentaje
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    # Obtener matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Obtener informe de clasificación como diccionario
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Convertir métricas a porcentaje y formatear
    for key in report_dict.keys():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            report_dict[key]['precision'] = report_dict[key]['precision'] * 100
            report_dict[key]['recall'] = report_dict[key]['recall'] * 100
            report_dict[key]['f1-score'] = report_dict[key]['f1-score'] * 100
        elif key in ['macro avg', 'weighted avg']:
            report_dict[key]['precision'] = report_dict[key]['precision'] * 100
            report_dict[key]['recall'] = report_dict[key]['recall'] * 100
            report_dict[key]['f1-score'] = report_dict[key]['f1-score'] * 100
    
    # Formatear informe de clasificación para impresión
    report_formatted = "Informe de clasificación:\n"
    report_formatted += "{:<15} {:>15} {:>15} {:>15} {:>10}\n".format("Clase", "Precisión (%)", "Recall (%)", "F1-Score (%)", "Soporte")
    for key in report_dict.keys():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            support = int(report_dict[key]['support'])
            precision = report_dict[key]['precision']
            recall = report_dict[key]['recall']
            f1_score = report_dict[key]['f1-score']
            report_formatted += "{:<15} {:>15.2f} {:>15.2f} {:>15.2f} {:>10}\n".format(key, precision, recall, f1_score, support)
    # Añadir macro avg y weighted avg
    for avg in ['macro avg', 'weighted avg']:
        precision = report_dict[avg]['precision']
        recall = report_dict[avg]['recall']
        f1_score = report_dict[avg]['f1-score']
        support = int(report_dict[avg]['support'])
        report_formatted += "{:<15} {:>15.2f} {:>15.2f} {:>15.2f} {:>10}\n".format(avg, precision, recall, f1_score, support)
    
    # Añadir precisión global
    overall_accuracy = accuracy_score(y_test, y_pred) * 100
    report_formatted += "\nPrecisión global (Accuracy): {:.2f}%\n".format(overall_accuracy)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report_formatted
    }
    return results
