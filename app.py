import os
import PyPDF2
from sklearn.feature_extraction.text import TFIDFVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


# Baixar recursos do NLTK (apenas na primeira execução)
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
vectorizer = TFIDFVectorizer()
X = vectorizer.fit_transform(textos)

print("Matriz BoW: ", X.toarray())
print("Vocabulário: ", vectorizer.get_feature_names_out())
print("Classes: ", classes)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)

# Modelos de algoritmos de aprendizado supervisionado
modelos = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'MLP': MLPClassifier(max_iter=1000)
}

# Validação cruzada e avaliação
resultados = {}

print("\nIniciando validação cruzada com 10 folds...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    print(f"\nTreinando modelo: {nome}")
    acuracias = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='f1_weighted')

    resultados[nome] = {
        'Acurácia Média': np.mean(acuracias),
        'Acurácia Desvio Padrão': np.std(acuracias),
        'F1-Score Médio': np.mean(f1_scores),
        'F1-Score Desvio Padrão': np.std(f1_scores),
    }

    print(f"Acurácia Média: {resultados[nome]['Acurácia Média']:.4f}")
    print(f"F1-Score Médio: {resultados[nome]['F1-Score Médio']:.4f}")

# Exibindo resultados
print("\nResultados Finais:")
for modelo, metricas in resultados.items():
    print(f"\nModelo: {modelo}")
    for metrica, valor in metricas.items():
        print(f"{metrica}: {valor:.4f}")
