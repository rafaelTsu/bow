import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np


# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Fun¸c~ao para extrair texto de PDFs
def pdf_para_txt(caminho_pdf):
    with open(caminho_pdf, "rb") as f:
        leitor = PyPDF2.PdfReader(f)
        texto = ""
        for pagina in range(len(leitor.pages)):
            texto += leitor.pages[pagina].extract_text()
    return texto

# Diretórios com os PDFs
diretorios = {
    'poesia': 'pdfs/poesia/',
    'prosa': 'pdfs/prosa/',
    'jornalismo': 'pdfs/jornalismo/'
}

# Função para limpar e remover stopwords
def limpar_texto(texto):
    stop_words = set(stopwords.words("english"))
    palavras = word_tokenize(texto.lower())
    palavras_limpa = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]
    return " ".join(palavras_limpa)

# Extraindo textos e gerando classes
textos = []
classes = []

for classe, caminho in diretorios.items():
    for arquivo in os.listdir(caminho):
        if arquivo.endswith(".pdf"):
            texto = pdf_para_txt(os.path.join(caminho, arquivo))
            texto_limpo = limpar_texto(texto)
            textos.append(texto_limpo)
            classes.append(classe)


# Criando a matriz Bag of Words
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

print("Matriz BoW: ", X.toarray())
print("Vocabulário: ", vectorizer.get_feature_names_out())
print("Classes: ", classes)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)

# Função para avaliar modelos com validação cruzada
def avalia_modelo(modelo, X, y):
    skf = StratifiedKFold(n_splits=10)
    acuracias = cross_val_score(modelo, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(modelo, X, y, cv=skf, scoring='f1_macro')
    print(f"Acurácia média: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
    print(f"F1-Score médio: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")

# Algoritmos de aprendizado supervisionado
algoritmos = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(max_iter=500)
}

# Avaliação de cada modelo
for nome, modelo in algoritmos.items():
    print(f"\nModelo: {nome}")
    avalia_modelo(modelo, X, y)