from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def train_knn(X_train, y_train):
    """
    Treina um modelo K-Nearest Neighbors (KNN) usando validação cruzada de 10 folds.
    """
    modelo = KNeighborsClassifier(n_neighbors=5)  # Pode ajustar 'n_neighbors' conforme necessário
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    acuracias = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='f1_weighted')

    print("\nResultados do KNN:")
    print(f"Acurácia Média: {np.mean(acuracias):.4f}")
    print(f"F1-Score Médio: {np.mean(f1_scores):.4f}")
