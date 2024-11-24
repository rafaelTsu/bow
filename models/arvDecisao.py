from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def train_arvore_decisao(X_train, y_train):
    """
    Treina um modelo de Árvore de Decisão usando validação cruzada de 10 folds.
    """
    modelo = DecisionTreeClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    acuracias = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='f1_weighted')

    print("\nResultados da Árvore de Decisão:")
    print(f"Acurácia Média: {np.mean(acuracias):.4f}")
    print(f"F1-Score Médio: {np.mean(f1_scores):.4f}")
