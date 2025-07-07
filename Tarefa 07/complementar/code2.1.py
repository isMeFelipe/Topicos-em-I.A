import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def word_tokenize_manual(text):
    return re.findall(r'\b\w+\b', text)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize_manual(text)
    stop_words = set(stopwords.words('portuguese'))  # Ajuste se o texto for em português
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def plot_confusion(y_test, y_pred, model_name, ax):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Matriz de Confusão - {model_name}")
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')

def evaluate_model(model, X_test, y_test, model_name, ax=None):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    minority_class = min(classes, key=lambda x: report[x]['support'])

    results = {
        'Modelo': model_name,
        'Acurácia': accuracy,
        'Precisão (minoritária)': report[minority_class]['precision'],
        'Revocação (minoritária)': report[minority_class]['recall'],
        'F1 (minoritária)': report[minority_class]['f1-score'],
        'Técnica Vetorização': model_name.split('(')[1][:-1]
    }

    if ax:
        plot_confusion(y_test, y_pred, model_name, ax)

    return results

def main(csv_file):
    df = pd.read_csv(csv_file)

    if len(df.columns) < 2:
        print("Erro: O dataset deve conter pelo menos duas colunas (rótulo e texto).")
        return

    X = df.iloc[:, 1]
    y = df.iloc[:, 0]

    print("\nPré-processando textos...")
    X_processed = X.apply(preprocess_text)

    print("\nDividindo dados em treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    results = []

    # Bag of Words
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Embeddings
    print("\nGerando embeddings com SentenceTransformer...")
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    X_train_embed = embedder.encode(X_train.tolist(), convert_to_tensor=False)
    X_test_embed = embedder.encode(X_test.tolist(), convert_to_tensor=False)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()

    # Modelos BoW
    print("\nTreinando modelos com Bag of Words...")
    svm_bow = SVC(kernel='linear')
    svm_bow.fit(X_train_bow, y_train)
    results.append(evaluate_model(svm_bow, X_test_bow, y_test, "SVM (BoW)", ax=axes[0]))

    rf_bow = RandomForestClassifier()
    rf_bow.fit(X_train_bow, y_train)
    results.append(evaluate_model(rf_bow, X_test_bow, y_test, "Random Forest (BoW)", ax=axes[1]))

    # Modelos TF-IDF
    print("\nTreinando modelos com TF-IDF...")
    svm_tfidf = SVC(kernel='linear')
    svm_tfidf.fit(X_train_tfidf, y_train)
    results.append(evaluate_model(svm_tfidf, X_test_tfidf, y_test, "SVM (TF-IDF)", ax=axes[2]))

    rf_tfidf = RandomForestClassifier()
    rf_tfidf.fit(X_train_tfidf, y_train)
    results.append(evaluate_model(rf_tfidf, X_test_tfidf, y_test, "Random Forest (TF-IDF)", ax=axes[3]))

    # Modelos com Embeddings
    print("\nTreinando modelos com Embeddings...")
    svm_emb = SVC(kernel='linear')
    svm_emb.fit(X_train_embed, y_train)
    results.append(evaluate_model(svm_emb, X_test_embed, y_test, "SVM (Embeddings)", ax=axes[4]))

    rf_emb = RandomForestClassifier()
    rf_emb.fit(X_train_embed, y_train)
    results.append(evaluate_model(rf_emb, X_test_embed, y_test, "Random Forest (Embeddings)", ax=axes[5]))

    print("\nResultados Comparativos:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main("comentarios1.csv")
