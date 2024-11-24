import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Módulos adicionados por você
from models.arvDecisao import train_arvore_decisao  # Adicionado
from models.knn import train_knn  # Adicionado

# Baixar recursos do NLTK (apenas na primeira execução)
nltk.download('punkt')
nltk.download('stopwords')

# Função para extrair texto de PDFs
def pdf_para_txt(caminho_pdf):
    with open(caminho_pdf, "rb") as f:
        leitor = PyPDF2.PdfReader(f)
        texto = ""
        for pagina in range(len(leitor.pages)):
            texto += leitor.pages[pagina].extract_text()
    return texto

# Função para limpar texto e remover stopwords
def limpar_texto(texto):
    stop_words = set(stopwords.words("portuguese"))  # Alterado para português
    palavras = word_tokenize(texto.lower())
    palavras_limpa = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]
    return " ".join(palavras_limpa)

# Diretórios com os PDFs
diretorios = {
    'poesia': 'pdfs/poesia/',
    'prosa': 'pdfs/prosa/',
    'jornalismo': 'pdfs/jornalismo/'
}

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

# Criando a matriz TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

print("Matriz TF-IDF criada com sucesso!")
print("Vocabulário: ", vectorizer.get_feature_names_out())
print("Classes: ", classes)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)

# Minhas modificações: Chamando os modelos de treinamento
print("Treinando modelo: Árvore de Decisão")  # Adicionado
train_arvore_decisao(X_train, y_train)  # Adicionado

print("\nTreinando modelo: KNN")  # Adicionado
train_knn(X_train, y_train)  # Adicionado
