import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score
import torch

# Carregar os dados de teste
test_df = pd.read_json("data/test.jsonl", lines=True)

# Filtrar apenas as colunas relevantes (text e label)
test_df = test_df[['text', 'label']]

# Converter o DataFrame pandas para o formato Dataset do Hugging Face
test_data = Dataset.from_pandas(test_df)

# Carregar o modelo e o tokenizador treinado
model = AutoModelForSequenceClassification.from_pretrained("./models/hatebert_model")
tokenizer = AutoTokenizer.from_pretrained("./models/hatebert_model")

# Função para tokenizar os dados
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenizar os dados
test_data = test_data.map(tokenize_function, batched=True)

# Função para prever e calcular a acurácia
def compute_metrics(p):
    preds = torch.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Configurar a avaliação
trainer = Trainer(
    model=model,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Avaliar o modelo
results = trainer.evaluate()

# Exibir os resultados
print("Acurácia no conjunto de teste:", results['eval_accuracy'])
