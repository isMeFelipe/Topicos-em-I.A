from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.utils.analyze import (
    extract_video_info,
    extract_comments,
    translate_texts,
    analyze_comments,
    analyze_temporal,
    detect_political_context,
)
import joblib
from app.utils.emotion_model import get_emotion_scores
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# Definir os limiares fixos para cada parâmetro
PARAM_THRESHOLD = {
    'clickbait': 0.8,        # Limiar para a pontuação de clickbait
    'hate_speech': 0.7,      # Limiar para a pontuação de discurso de ódio
    'temporal': 0.5,         # Limiar para a análise temporal
    'political': 0.5,        # Limiar para o contexto político
    'anger': 0.5             # Limiar para a pontuação de raiva
}


def classify_ragebait(scores):
    """
    Função para classificar com base em limiares fixos.
    """
    # A classificação como 'Ragebait' será baseada na soma de todas as pontuações
    score_sum = sum([scores[param] >= threshold for param, threshold in PARAM_THRESHOLD.items()])
    
    # Se a soma dos parâmetros acima do limiar for maior que um valor específico, é "Ragebait"
    return "Ragebait" if score_sum >= 4 else "Não Ragebait"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_video(request: Request, video_url: str = Form(...)):
    video_info = extract_video_info(video_url)
    title, description, publish_time = video_info['title'], video_info['description'], video_info['publish_time']
    text = f"{title} {description}"

    # Carregando modelos
    clickbait_model = joblib.load('models/clickbait_model.pkl')

    HATEBERT_PATH = "models/hatebert_model"

    hatebert_model = AutoModelForSequenceClassification.from_pretrained(HATEBERT_PATH)
    hatebert_tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')
    hatebert_model.eval()

    # Calcular as pontuações dos modelo
    clickbait_score = clickbait_model.predict_proba([text])[0][1]
    comments = extract_comments(video_url, max_comments=10000)

    hate_speech_score = analyze_comments(comments, hatebert_model, hatebert_tokenizer)
    
    temporal_score = analyze_temporal(comments, publish_time)
    political_score = detect_political_context(text)
    emotion_scores = get_emotion_scores(text)
    anger_score = emotion_scores.get("anger", 0.0)

    # Armazenar as pontuações em um dicionário
    scores = {
        'clickbait': clickbait_score,
        'hate_speech': hate_speech_score,
        'temporal': temporal_score,
        'political': political_score,
        'anger': anger_score
    }

    # Classificar baseado nos limiares
    ragebait_classification = classify_ragebait(scores)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "video_url": video_url,
        "clickbait_score": f"{clickbait_score:.2f}",
        "hate_speech_score": f"{hate_speech_score:.2f}",
        "temporal_score": f"{temporal_score:.2f}",
        "political_score": f"{political_score:.2f}",
        "anger_score": f"{anger_score:.2f}",
        "ragebait_classification": ragebait_classification
    })