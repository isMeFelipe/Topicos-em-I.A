import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_clickbait_model(path="models/clickbait_model.pkl"):
    return joblib.load(path)

def load_hatebert_model(path="models/hatebert_model"):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')
    model.eval()
    return model, tokenizer
