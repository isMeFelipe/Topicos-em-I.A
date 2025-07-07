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
    # Tenta identificar se é um vídeo do tipo Shorts
    if "youtube.com/shorts/" in video_url:
        video_id = video_url.split("shorts/")[1].split("?")[0]  # Extrai o ID do Shorts
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = video_url.split("v=")[1].split("&")[0]  # Para vídeos padrão

    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url).json()

    video_info = {
        "title": response["items"][0]["snippet"]["title"],
        "description": response["items"][0]["snippet"]["description"],
        "publish_time": response["items"][0]["snippet"]["publishedAt"]
    }

    return video_info


# Função para extrair os comentários de um vídeo
def extract_comments(video_url: str, max_comments: int = 10000):
    # Verifique se a URL é de um vídeo Shorts
    if "youtube.com/shorts/" in video_url:
        try:
            video_id = video_url.split("shorts/")[1].split("?")[0]
        except IndexError:
            raise ValueError(f"Falha ao extrair o ID do Shorts da URL: {video_url}")
    # Verifique se a URL é de um vídeo curto (youtu.be)
    elif "youtu.be/" in video_url:
        try:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        except IndexError:
            raise ValueError(f"Falha ao extrair o ID do vídeo da URL: {video_url}")
    # Para vídeos padrão (youtube.com/watch?v=ID)
    else:
        try:
            video_id = video_url.split("v=")[1].split("&")[0]
        except IndexError:
            raise ValueError(f"Falha ao extrair o ID do vídeo da URL: {video_url}")
        
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

# Função para detectar o contexto político
def detect_political_context(text: str) -> float:
    # Aqui você pode implementar um modelo de detecção de contexto político ou usar alguma abordagem de NLP
    # Exemplo fictício: se o texto contiver certas palavras-chave relacionadas a política, retorna uma pontuação mais alta
    political_keywords = ["election", "government", "politics", "policy", "vote"]
    political_score = sum([1 for word in political_keywords if word in text.lower()])
    
    # Normalizando a pontuação
    return political_score / len(political_keywords)