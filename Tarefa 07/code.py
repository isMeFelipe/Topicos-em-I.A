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
    except FileNotFoundError:
        print(f"Erro: Arquivo '{csv_file}' não encontrado.")
        return None
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
    
    print("\nPrimeiras linhas do dataset original:")
    print(df.head())
    
    X = df.iloc[:, 1]
    y = df.iloc[:, 0]

    print("\nPré-processando os textos...")
    X_processed = X.apply(preprocess_text)

    print("\nExemplo de texto antes e depois do pré-processamento:")
    print("Antes:", X.iloc[0])
    print("Depois:", X_processed.iloc[0])

    print("\nDistribuição das classes:")
    print(y.value_counts())

    X_train = X_processed
    y_train = y

    print("\nCriando vetores Bag of Words (Uni-gramas)...")
    bow_vectorizer = CountVectorizer()
    X_train_uni = bow_vectorizer.fit_transform(X_train)

    print("\nCriando vetores TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    all_results = []

    print("\nTreinando modelos com Bag of Words (Uni-gramas)...")
    svm_bow = SVC(kernel='linear', random_state=42)
    svm_bow.fit(X_train_uni, y_train)
    all_results.append(evaluate_model(svm_bow, X_train_uni, y_train, "SVM (BoW)", is_test_data=False))

    rf_bow = RandomForestClassifier(random_state=42)
    rf_bow.fit(X_train_uni, y_train)
    all_results.append(evaluate_model(rf_bow, X_train_uni, y_train, "Random Forest (BoW)", is_test_data=False))

    print("\nTreinando modelos com TF-IDF...")
    svm_tfidf = SVC(kernel='linear', random_state=42)
    svm_tfidf.fit(X_train_tfidf, y_train)
    all_results.append(evaluate_model(svm_tfidf, X_train_tfidf, y_train, "SVM (TF-IDF)", is_test_data=False))

    rf_tfidf = RandomForestClassifier(random_state=42)
    rf_tfidf.fit(X_train_tfidf, y_train)
    all_results.append(evaluate_model(rf_tfidf, X_train_tfidf, y_train, "Random Forest (TF-IDF)", is_test_data=False))

    print("\nResultados comparativos:")
    print(tabulate(all_results, headers="keys", tablefmt="grid"))

    return {
        'X_train_uni': X_train_uni,
        'X_train_tfidf': X_train_tfidf,
        'y_train': y_train,
        'models': {
            'svm_bow': svm_bow,
            'rf_bow': rf_bow,
            'svm_tfidf': svm_tfidf,
            'rf_tfidf': rf_tfidf
        },
        'vectorizers': {
            'bow': bow_vectorizer,
            'tfidf': tfidf_vectorizer
        },
        'results': all_results
    }

def prepare_test_data(csv_file, vectorizers):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, 1]
    y = df.iloc[:, 0]
    X_processed = X.apply(preprocess_text)

    X_test_uni = vectorizers['bow'].transform(X_processed)
    X_test_tfidf = vectorizers['tfidf'].transform(X_processed)

    return {
        'X_train_uni': X_test_uni,
        'X_train_tfidf': X_test_tfidf,
        'y_train': y
    }

if __name__ == "__main__":
    arquivo_train = "train.csv"
    resultado_train = load_and_prepare_data(arquivo_train, 100000)

    if resultado_train:
        print("\nTreinamento concluído com sucesso!")
        print("\nResumo das dimensões dos dados de treino:")
        print(f"- Vetores BoW: {resultado_train['X_train_uni'].shape[1]} features")
        print(f"- Vetores TF-IDF: {resultado_train['X_train_tfidf'].shape[1]} features")
        print(f"- Exemplos de treino: {resultado_train['X_train_uni'].shape[0]}")

    arquivo_test = "test.csv"
    resultado_test = prepare_test_data(arquivo_test, resultado_train['vectorizers'])

    if resultado_test:
        print("\nAnálise no conjunto de testes concluída com sucesso!")

        all_results_test = []
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        print("\nAvaliação no conjunto de teste com Bag of Words (BoW)...")
        all_results_test.append(evaluate_model(resultado_train['models']['svm_bow'], resultado_test['X_train_uni'], resultado_test['y_train'], "SVM (BoW)", is_test_data=True, ax=axes[0]))
        all_results_test.append(evaluate_model(resultado_train['models']['rf_bow'], resultado_test['X_train_uni'], resultado_test['y_train'], "Random Forest (BoW)", is_test_data=True, ax=axes[1]))

        print("\nAvaliação no conjunto de teste com TF-IDF...")
        all_results_test.append(evaluate_model(resultado_train['models']['svm_tfidf'], resultado_test['X_train_tfidf'], resultado_test['y_train'], "SVM (TF-IDF)", is_test_data=True, ax=axes[2]))
        all_results_test.append(evaluate_model(resultado_train['models']['rf_tfidf'], resultado_test['X_train_tfidf'], resultado_test['y_train'], "Random Forest (TF-IDF)", is_test_data=True, ax=axes[3]))

        print("\nResultados de avaliação no conjunto de testes:")
        print(tabulate(all_results_test, headers="keys", tablefmt="grid"))

        plt.tight_layout()
        plt.show()
