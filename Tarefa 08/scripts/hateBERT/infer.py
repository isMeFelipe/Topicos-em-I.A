from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Carregar o modelo e tokenizador treinado
model = AutoModelForSequenceClassification.from_pretrained("./models/hatebert_model")
tokenizer = AutoTokenizer.from_pretrained("./models/hatebert_model")

# Função para prever discurso de ódio
def predict_hate_speech(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()

# Exemplo de uso
comment = "I hate this!"
prediction = predict_hate_speech(comment)

if prediction == 1:  # Discurso de ódio
    print("Discurso de ódio detectado")
else:
    print("Não é discurso de ódio")
