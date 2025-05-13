import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import List, Tuple
import torch

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def translate_texts(texts, model, tokenizer, batch_size=8):
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=512)
        translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated.extend(translated_batch)
    return translated

# Função para extrair informações do vídeo
def extract_video_info(video_url: str) -> dict:
    video_id = video_url.split("v=")[1].split("&")[0]
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url).json()

    video_info = {
        "title": response["items"][0]["snippet"]["title"],
        "description": response["items"][0]["snippet"]["description"],
        "publish_time": response["items"][0]["snippet"]["publishedAt"]
    }

    return video_info

# Função para extrair os comentários de um vídeo
def extract_comments(video_url: str, max_comments: int = 100) -> List[str]:
    video_id = video_url.split("v=")[1].split("&")[0]
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults={max_comments}&key={YOUTUBE_API_KEY}"
    response = requests.get(url).json()

    comments = [
        item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for item in response.get("items", [])
    ]

    return comments

# Função para analisar os comentários (sentimento e discurso de ódio)
def analyze_comments(comments, model, tokenizer):
    import torch
    from torch.nn.functional import softmax

    hate_speech_scores = []

    for comment in comments:
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        hate_score = probs[0][1].item()  # classe 1 = hate speech
        hate_speech_scores.append(hate_score)

    avg_hate = sum(hate_speech_scores) / len(hate_speech_scores) if hate_speech_scores else 0.0
    return avg_hate



# Função para análise temporal (por exemplo, a idade do vídeo)
def analyze_temporal(comments: list[str], publish_time: str) -> float:
    try:
        publish_datetime = datetime.strptime(publish_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        publish_datetime = datetime.strptime(publish_time, "%Y-%m-%dT%H:%M:%SZ")
    
    current_datetime = datetime.now()
    time_diff = (current_datetime - publish_datetime).days

    # Quanto mais recente, maior a pontuação (de 0 a 1)
    temporal_score = max(0, 1 - time_diff / 365)
    return temporal_score

# Função para detectar o contexto político
def detect_political_context(text: str) -> float:
    # Aqui você pode implementar um modelo de detecção de contexto político ou usar alguma abordagem de NLP
    # Exemplo fictício: se o texto contiver certas palavras-chave relacionadas a política, retorna uma pontuação mais alta
    political_keywords = ["election", "government", "politics", "policy", "vote"]
    political_score = sum([1 for word in political_keywords if word in text.lower()])
    
    # Normalizando a pontuação
    return political_score / len(political_keywords)