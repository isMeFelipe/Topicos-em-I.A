import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate
from sentence_transformers import SentenceTransformer


def word_tokenize_manual(text):
    return re.findall(r'\b\w+\b', text)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize_manual(text)
    stop_words = set(stopwords.words('english'))
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


def evaluate_model(model, X_test, y_test, model_name, is_test_data=False, ax=None):
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

    if is_test_data and ax:
        plot_confusion(y_test, y_pred, model_name, ax)

    return results


def load_and_prepare_data(csv_file, max_train_records=None):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

    if len(df.columns) < 2:
        print("Erro: O dataset deve ter pelo menos duas colunas (rótulo e texto).")
        return None

    if max_train_records:
        df = df.sample(n=min(max_train_records, len(df)), random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    X = df.iloc[:, 1]
    y = df.iloc[:, 0]

    print("\nPré-processando os textos...")
    X_processed = X.apply(preprocess_text)

    print("\nGerando Bag of Words e TF-IDF...")
    bow_vectorizer = CountVectorizer()
    X_train_uni = bow_vectorizer.fit_transform(X_processed)

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_processed)

    print("\nGerando embeddings com SentenceTransformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_train_emb = embedder.encode(X_processed.tolist(), show_progress_bar=True)

    all_results = []

    print("\nTreinando modelos com BoW...")
    svm_bow = SVC(kernel='linear', random_state=42)
    svm_bow.fit(X_train_uni, y)
    all_results.append(evaluate_model(svm_bow, X_train_uni, y, "SVM (BoW)"))

    rf_bow = RandomForestClassifier(random_state=42)
    rf_bow.fit(X_train_uni, y)
    all_results.append(evaluate_model(rf_bow, X_train_uni, y, "Random Forest (BoW)"))

    print("\nTreinando modelos com TF-IDF...")
    svm_tfidf = SVC(kernel='linear', random_state=42)
    svm_tfidf.fit(X_train_tfidf, y)
    all_results.append(evaluate_model(svm_tfidf, X_train_tfidf, y, "SVM (TF-IDF)"))

    rf_tfidf = RandomForestClassifier(random_state=42)
    rf_tfidf.fit(X_train_tfidf, y)
    all_results.append(evaluate_model(rf_tfidf, X_train_tfidf, y, "Random Forest (TF-IDF)"))

    print("\nTreinando modelos com Embeddings...")
    svm_emb = SVC(kernel='linear', random_state=42)
    svm_emb.fit(X_train_emb, y)
    all_results.append(evaluate_model(svm_emb, X_train_emb, y, "SVM (Embeddings)"))

    rf_emb = RandomForestClassifier(random_state=42)
    rf_emb.fit(X_train_emb, y)
    all_results.append(evaluate_model(rf_emb, X_train_emb, y, "Random Forest (Embeddings)"))

    print("\nResultados comparativos:")
    print(tabulate(all_results, headers="keys", tablefmt="grid"))

    return {
        'X_train_uni': X_train_uni,
        'X_train_tfidf': X_train_tfidf,
        'X_train_emb': X_train_emb,
        'y_train': y,
        'models': {
            'svm_bow': svm_bow,
            'rf_bow': rf_bow,
            'svm_tfidf': svm_tfidf,
            'rf_tfidf': rf_tfidf,
            'svm_emb': svm_emb,
            'rf_emb': rf_emb
        },
        'vectorizers': {
            'bow': bow_vectorizer,
            'tfidf': tfidf_vectorizer
        },
        'embedder': embedder
    }


def prepare_test_data(csv_file, vectorizers, embedder):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, 1]
    y = df.iloc[:, 0]
    X_processed = X.apply(preprocess_text)

    X_test_uni = vectorizers['bow'].transform(X_processed)
    X_test_tfidf = vectorizers['tfidf'].transform(X_processed)
    X_test_emb = embedder.encode(X_processed.tolist(), show_progress_bar=True)

    return {
        'X_test_uni': X_test_uni,
        'X_test_tfidf': X_test_tfidf,
        'X_test_emb': X_test_emb,
        'y_test': y
    }


if __name__ == "__main__":
    train_file = "comentarios.csv"
    test_file = "comentarios1.csv"

    resultado_train = load_and_prepare_data(train_file, max_train_records=100000)

    if resultado_train:
        resultado_test = prepare_test_data(test_file, resultado_train['vectorizers'], resultado_train['embedder'])

        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        axes = axes.flatten()

        all_results_test = []
        all_results_test.append(evaluate_model(resultado_train['models']['svm_bow'], resultado_test['X_test_uni'], resultado_test['y_test'], "SVM (BoW)", is_test_data=True, ax=axes[0]))
        all_results_test.append(evaluate_model(resultado_train['models']['rf_bow'], resultado_test['X_test_uni'], resultado_test['y_test'], "Random Forest (BoW)", is_test_data=True, ax=axes[1]))
        all_results_test.append(evaluate_model(resultado_train['models']['svm_tfidf'], resultado_test['X_test_tfidf'], resultado_test['y_test'], "SVM (TF-IDF)", is_test_data=True, ax=axes[2]))
        all_results_test.append(evaluate_model(resultado_train['models']['rf_tfidf'], resultado_test['X_test_tfidf'], resultado_test['y_test'], "Random Forest (TF-IDF)", is_test_data=True, ax=axes[3]))
        all_results_test.append(evaluate_model(resultado_train['models']['svm_emb'], resultado_test['X_test_emb'], resultado_test['y_test'], "SVM (Embeddings)", is_test_data=True, ax=axes[4]))
        all_results_test.append(evaluate_model(resultado_train['models']['rf_emb'], resultado_test['X_test_emb'], resultado_test['y_test'], "Random Forest (Embeddings)", is_test_data=True, ax=axes[5]))

        print("\nResultados de avaliação no conjunto de testes:")
        print(tabulate(all_results_test, headers="keys", tablefmt="grid"))

        plt.tight_layout()
        plt.show()