from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Nome do modelo
model_name = "j-hartmann/emotion-english-distilroberta-base"

# Carregar tokenizador e modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Criar pipeline de classificação
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Texto de exemplo
text = "I hate everything about this!"

# Inferência
results = emotion_classifier(text)

# Exibir resultados ordenados por maior score
for res in sorted(results[0], key=lambda x: x["score"], reverse=True):
    print(f"{res['label']}: {res['score']:.4f}")
