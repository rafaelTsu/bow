from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def train_knn(X_train, y_train):
    """
    Treina um modelo K-Nearest Neighbors (KNN) usando validação cruzada de 10 folds
    e calcula a média e o desvio padrão das métricas.
    """
    modelo = KNeighborsClassifier(n_neighbors=5)  # Pode ajustar 'n_neighbors' conforme necessário
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Acurácia
    acuracias = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='accuracy')
    media_acuracia = np.mean(acuracias)
    desvio_acuracia = np.std(acuracias)

    # F1-Score
    f1_scores = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='f1_weighted')
    media_f1 = np.mean(f1_scores)
    desvio_f1 = np.std(f1_scores)

    print("\nResultados do KNN:")
    print(f"Acurácia Média: {media_acuracia:.4f}, Desvio Padrão: {desvio_acuracia:.4f}")
    print(f"F1-Score Médio: {media_f1:.4f}, Desvio Padrão: {desvio_f1:.4f}")
