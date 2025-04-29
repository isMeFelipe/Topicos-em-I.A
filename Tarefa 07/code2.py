import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Pré-processamento de texto"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def evaluate_model(model, X_test, y_test, model_name):
    """Avalia um modelo e retorna métricas"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Identificar classe minoritária
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    minority_class = min(classes, key=lambda x: report[x]['support'])
    
    return {
        'Modelo': model_name,
        'Acurácia': accuracy,
        'Precisão (minoritária)': report[minority_class]['precision'],
        'Revocação (minoritária)': report[minority_class]['recall'],
        'F1 (minoritária)': report[minority_class]['f1-score'],
        'Técnica Vetorização': model_name.split('(')[1][:-1]  # Extrai o tipo de vetorização
    }

def load_and_prepare_data(csv_file):
    """Função principal para carregar e preparar dados"""
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
    
    print("\nPrimeiras linhas do dataset original:")
    print(df.head())
    
    # Separar features e rótulos
    X = df.iloc[:, 1]  # Assumindo que a segunda coluna é o texto
    y = df.iloc[:, 0]   # Assumindo que a primeira coluna é o rótulo
    
    # Pré-processar textos
    print("\nPré-processando os textos...")
    X_processed = X.apply(preprocess_text)
    
    print("\nExemplo de texto antes e depois do pré-processamento:")
    print("Antes:", X.iloc[0])
    print("Depois:", X_processed.iloc[0])
    
    print("\nDistribuição das classes:")
    print(y.value_counts())
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Vetorização Bag of Words (Uni-gramas)
    print("\nCriando vetores Bag of Words (Uni-gramas)...")
    bow_vectorizer = CountVectorizer()
    X_train_uni = bow_vectorizer.fit_transform(X_train)
    X_test_uni = bow_vectorizer.transform(X_test)
    
    # Vetorização TF-IDF
    print("\nCriando vetores TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Lista para armazenar todos os resultados
    all_results = []
    
    # Treinar e avaliar modelos com BoW
    print("\nTreinando modelos com Bag of Words (Uni-gramas)...")
    
    # SVM com BoW
    svm_bow = SVC(kernel='linear', random_state=42)
    svm_bow.fit(X_train_uni, y_train)
    all_results.append(evaluate_model(svm_bow, X_test_uni, y_test, "SVM (BoW)"))
    
    # Random Forest com BoW
    rf_bow = RandomForestClassifier(random_state=42)
    rf_bow.fit(X_train_uni, y_train)
    all_results.append(evaluate_model(rf_bow, X_test_uni, y_test, "Random Forest (BoW)"))
    
    # Treinar e avaliar modelos com TF-IDF
    print("\nTreinando modelos com TF-IDF...")
    
    # SVM com TF-IDF
    svm_tfidf = SVC(kernel='linear', random_state=42)
    svm_tfidf.fit(X_train_tfidf, y_train)
    all_results.append(evaluate_model(svm_tfidf, X_test_tfidf, y_test, "SVM (TF-IDF)"))
    
    # Random Forest com TF-IDF
    rf_tfidf = RandomForestClassifier(random_state=42)
    rf_tfidf.fit(X_train_tfidf, y_train)
    all_results.append(evaluate_model(rf_tfidf, X_test_tfidf, y_test, "Random Forest (TF-IDF)"))
    
    # Exibir resultados comparativos
    print("\nResultados comparativos:")
    print(tabulate(all_results, headers="keys", tablefmt="grid"))
    
    # Separar resultados por técnica de vetorização para análise
    bow_results = [r for r in all_results if 'BoW' in r['Modelo']]
    tfidf_results = [r for r in all_results if 'TF-IDF' in r['Modelo']]
    
    print("\nComparação entre técnicas de vetorização:")
    print("\nMelhor Acurácia:")
    print(f"BoW: {max(r['Acurácia'] for r in bow_results):.4f}")
    print(f"TF-IDF: {max(r['Acurácia'] for r in tfidf_results):.4f}")
    
    print("\nMelhor F1-Score (classe minoritária):")
    print(f"BoW: {max(r['F1 (minoritária)'] for r in bow_results):.4f}")
    print(f"TF-IDF: {max(r['F1 (minoritária)'] for r in tfidf_results):.4f}")
    
    return {
        'X_train_uni': X_train_uni,
        'X_test_uni': X_test_uni,
        'X_train_tfidf': X_train_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'y_train': y_train,
        'y_test': y_test,
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

# Exemplo de uso
if __name__ == "__main__":
    arquivo_csv = "test.csv"  
    resultado = load_and_prepare_data(arquivo_csv)
    
    if resultado:
        print("\nProcesso concluído com sucesso!")
        print("\nResumo das dimensões dos dados:")
        print(f"- Vetores BoW: {resultado['X_train_uni'].shape[1]} features")
        print(f"- Vetores TF-IDF: {resultado['X_train_tfidf'].shape[1]} features")
        print(f"- Exemplos de treino: {resultado['X_train_uni'].shape[0]}")
        print(f"- Exemplos de teste: {resultado['X_test_uni'].shape[0]}")