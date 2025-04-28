import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

# Carregar o dataset
# Substitua pelo caminho do seu arquivo CSV
df = pd.read_csv("test.csv", header=None)
df = df.sample(n=2000, random_state=42)
print(df.columns)


# Definir os nomes das colunas
df.columns = ['index', 'titulo', 'texto']


# Exemplo de visualização de dados
print(df.head())

# Preprocessamento de dados
# A coluna 'texto' contém o texto, e a coluna 'classe' contém os rótulos
X = df['texto']  # Atributos
y = df['titulo']  # Rótulo

# Dividir o dataset em treino e teste (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Representação 1: Frequência de Uni-gramma (Count Vectorizer)
count_vectorizer = CountVectorizer()

# Representação 2: TF-IDF (Term Frequency - Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer()

# Função para treinar e avaliar um modelo (RF ou SVM) com diferentes vetores
def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer, model):
    # Criar um pipeline com a vetorização e o modelo
    pipeline = make_pipeline(vectorizer, model)
    
    # Treinar o modelo
    pipeline.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = pipeline.predict(X_test)
    
    # Avaliar o desempenho
    print(f"Desempenho para {model.__class__.__name__} com {vectorizer.__class__.__name__}:")
    print(classification_report(y_test, y_pred))
    print("="*80)

# Random Forest (RF)
rf = RandomForestClassifier(random_state=42)

# Support Vector Machine (SVM)
svm = SVC(random_state=42)

# Treinar e avaliar com a representação de Uni-gramma (Count Vectorizer)
train_and_evaluate(X_train, X_test, y_train, y_test, count_vectorizer, rf)
train_and_evaluate(X_train, X_test, y_train, y_test, count_vectorizer, svm)

# Treinar e avaliar com a representação TF-IDF
train_and_evaluate(X_train, X_test, y_train, y_test, tfidf_vectorizer, rf)
train_and_evaluate(X_train, X_test, y_train, y_test, tfidf_vectorizer, svm)
