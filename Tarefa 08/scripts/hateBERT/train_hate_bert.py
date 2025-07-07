import os
import jsonlines
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Verificando o diretório atual e informações do PyTorch
print(f"Diretório atual: {os.getcwd()}")
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

# Caminho do arquivo JSONL
file_path = "data/hateBERT/train.jsonl"

# Ler o arquivo usando jsonlines
with jsonlines.open(file_path) as reader:
    data = [obj for obj in reader]

# Converter para DataFrame e manter apenas colunas relevantes
train_df = pd.DataFrame(data)[['text', 'label']]
train_df = train_df[train_df['label'].isin([0, 1])]

# Verificar os primeiros registros
print(train_df.head())

# Converter o DataFrame pandas para o formato Dataset do Hugging Face
train_data = Dataset.from_pandas(train_df)

# Nome do modelo
model_name = "GroNLP/hateBERT"

# Carregar o tokenizador e o modelo pré-treinado
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Função de tokenização
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Aplicar tokenização
train_data = train_data.map(tokenize_function, batched=True)

# Remover colunas que não são usadas pelo modelo
train_data = train_data.remove_columns([col for col in train_data.column_names if col not in ["input_ids", "attention_mask", "label"]])

# Configurações de treinamento
training_args = TrainingArguments(
    output_dir="./models/hatebert_model",    # Diretório de saída do modelo treinado
    eval_strategy="no",                      # Estratégia de avaliação
    learning_rate=2e-5,                      # Taxa de aprendizado
    per_device_train_batch_size=8,           # Tamanho do batch por dispositivo
    num_train_epochs=3,                      # Número de épocas
    weight_decay=0.01,                       # Decaimento de peso
    use_cpu=True                             # Força o uso de CPU
)

# Inicializar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Treinar o modelo
trainer.train()

# Salvar o modelo treinado
trainer.save_model("./models/hatebert_model")
