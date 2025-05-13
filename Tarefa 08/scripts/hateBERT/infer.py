import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# Caminho do modelo treinado
model_path = "./models/hatebert_model"

# Carrega modelo e tokenizador
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')

# Coloca o modelo em modo de avaliação
model.eval()

# Exemplo de texto para testar
text = "kids like that disgust me!"

# Tokeniza o texto
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Desativa gradientes (modo de inferência)
with torch.no_grad():
    outputs = model(**inputs)

# Obtém as logits (valores antes do softmax)
logits = outputs.logits

# Aplica softmax para obter probabilidades
probs = softmax(logits, dim=-1)

# Classe prevista
predicted_class = torch.argmax(probs, dim=1).item()

print(f"Texto: {text}")
print(f"Classe prevista: {predicted_class} | Probabilidades: {probs.tolist()}")
