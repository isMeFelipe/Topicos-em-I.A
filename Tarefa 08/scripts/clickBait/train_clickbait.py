import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json

# Caminhos dos arquivos
INSTANCES_PATH = "data/instances.jsonl"
TRUTH_PATH = "data/truth.jsonl"
MODEL_OUTPUT_PATH = "models/clickbait_model.pkl"

def load_dataset(instances_path, truth_path):
    print("🔄 Carregando e mesclando os dados...")

    # Carregar os arquivos
    instances = [json.loads(line) for line in open(instances_path, 'r', encoding='utf-8')]
    truth = [json.loads(line) for line in open(truth_path, 'r', encoding='utf-8')]

    # Transformar em DataFrames
    df_instances = pd.DataFrame(instances)
    df_truth = pd.DataFrame(truth)

    # Mesclar pelo 'id'
    df = pd.merge(df_instances, df_truth, on="id")

    # Combinar textos relevantes (postText + targetTitle)
    df['text'] = df['postText'].apply(lambda x: x[0]) + " " + df['targetTitle']

    # Rótulos binários
    df['label'] = df['truthClass'].map({'clickbait': 1, 'no-clickbait': 0})

    return df[['text', 'label']]

def train_model(df):
    print("🧠 Treinando modelo...")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Avaliação
    y_pred = pipeline.predict(X_test)
    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Acurácia: {accuracy_score(y_test, y_pred):.2f}")

    return pipeline

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"💾 Modelo salvo em: {output_path}")

if __name__ == "__main__":
    df = load_dataset(INSTANCES_PATH, TRUTH_PATH)
    model = train_model(df)
    save_model(model, MODEL_OUTPUT_PATH)
