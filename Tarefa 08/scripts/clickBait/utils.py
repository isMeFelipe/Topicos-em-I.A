import os
from googleapiclient.discovery import build
from transformers import pipeline
from datetime import datetime
from dotenv import load_dotenv
import re

# Configurar API do YouTube

load_dotenv()

# Agora você pode acessar as variáveis de ambiente
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Pipeline de análise de sentimentos
sentiment_analyzer = pipeline("sentiment-analysis")

def extract_video_info(video_url):
    # Extrair ID do vídeo
    video_id = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id:
        raise ValueError("URL inválida do YouTube.")
    video_id = video_id.group(1)

    # Obter informações do vídeo
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    snippet = response['items'][0]['snippet']
    title = snippet['title']
    description = snippet['description']
    publish_time = snippet['publishedAt']

    return {
        'title': title,
        'description': description,
        'publish_time': publish_time
    }

def extract_comments(video_url, max_comments=10000):
    # Extrair ID do vídeo
    video_id = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id:
        raise ValueError("URL inválida do YouTube.")
    video_id = video_id.group(1)

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

    while response and len(comments) < max_comments:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100
            )
            response = request.execute()
        else:
            break

    return comments

def analyze_comments(comments, hatebert_model):
    # Análise de sentimentos
    sentiments = sentiment_analyzer(comments)
    negative_comments = [s for s in sentiments if s['label'] == 'NEGATIVE']
    sentiment_score = len(negative_comments) / len(comments) if comments else 0

    # Análise de discurso de ódio
    hate_speech_count = 0
    for comment in comments:
        # Aqui você deve implementar a inferência usando o modelo HateBERT
        # Exemplo:
        # inputs = tokenizer(comment, return_tensors="pt")
        # outputs = hatebert_model(**inputs)
        # prediction = torch.argmax(outputs.logits, dim=1)
        # if prediction == 1:
        #     hate_speech_count += 1
        pass  # Substitua pelo código real

    hate_speech_score = hate_speech_count / len(comments) if comments else 0

    return sentiment_score, hate_speech_score

def analyze_temporal(comments, publish_time):
    # Implementar análise temporal com base nas datas dos comentários
    # Exemplo: calcular a densidade de comentários nas primeiras 24 horas
    # Necessário obter as datas dos comentários na função extract_comments
    return 0.0  # Placeholder

def detect_political_context(text):
    political_keywords = ['política', 'governo', 'presidente', 'eleição', 'congresso', 'senado', 'deputado']
    count = sum(1 for word in political_keywords if word.lower() in text.lower())
    return count / len(political_keywords)

def compute_ragebait_score(clickbait_score, sentiment_score, hate_speech_score, temporal_score, political_score):
    # Combinar as pontuações com pesos
    weights = {
        'clickbait': 0.3,
        'sentiment': 0.2,
        'hate_speech': 0.2,
        'temporal': 0.1,
        'political': 0.2
    }
    score = (
        clickbait_score * weights['clickbait'] +
        sentiment_score * weights['sentiment'] +
        hate_speech_score * weights['hate_speech'] +
        temporal_score * weights['temporal'] +
        political_score * weights['political']
    )
    return score

