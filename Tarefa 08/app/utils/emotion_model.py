from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def get_emotion_scores(text):
    results = emotion_classifier(text)[0]
    return {item["label"].lower(): item["score"] for item in results}
