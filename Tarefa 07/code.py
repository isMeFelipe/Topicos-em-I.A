import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Carregar o dataset
# Substitua o nome do seu arquivo aqui
df = pd.read_csv('test.csv', header=None, names=['classe', 'texto'])
df = df.sample(n=500, random_state=42)

# 2. Separar atributos (X) e rótulos (y)
X = df['texto']
y = df['classe']

# 3. Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Função para treinar e avaliar modelos
def treinar_e_avaliar(X_train_vec, X_test_vec, nome_repr):
    # Modelos
    modelos = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }
    
    for nome_modelo, modelo in modelos.items():
        print(f"\n\n======= {nome_modelo} com {nome_repr} =======")
        modelo.fit(X_train_vec, y_train)
        y_pred = modelo.predict(X_test_vec)
        
        # Acurácia geral
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        
        # Relatório completo (Precisão, Revocação, F1 para cada classe)
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, digits=4))

# 5. Representação: Bag of Words (Frequência de unigramas)
print("=== Usando Bag of Words ===")
vectorizer_bow = CountVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

treinar_e_avaliar(X_train_bow, X_test_bow, "Bag of Words")

# 6. Representação: TF-IDF
print("\n=== Usando TF-IDF ===")
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

treinar_e_avaliar(X_train_tfidf, X_test_tfidf, "TF-IDF")
