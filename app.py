import nltk
from nltk.corpus import stopwords

# Baixar recursos do NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')

import PyPDF2

# Função para extrair texto de PDFs
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
    texto = texto.replace("\n", " ")
    stop_words = set(stopwords.words("english"))
    palavras = nltk.word_tokenize(texto.lower())
    palavras_limpa = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]
    return " ".join(palavras_limpa)


import os

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

from sklearn.feature_extraction.text import TfidfVectorizer

# Criando a matriz Bag of Words
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

print("Matriz BoW: ", X.toarray())
print("Vocabulário: ", vectorizer.get_feature_names_out())
print("Classes: ", classes)

from sklearn.model_selection import train_test_split

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)