from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.utils.analyze import (
    extract_video_info,
    extract_comments,
    translate_texts,
    analyze_comments,
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

PARAM_WEIGHTS = {
    'clickbait': 1.5,
    'anger': 1.5,
    'hate_speech': 2.0
}

def classify_ragebait(scores):
    weighted_score = sum([
        scores['clickbait'] * PARAM_WEIGHTS['clickbait'],
        scores['anger'] * PARAM_WEIGHTS['anger'],
        scores['hate_speech'] * PARAM_WEIGHTS['hate_speech']
    ])

    if weighted_score >= 2.8:
        return "Ragebait"
    return "Não Ragebait"




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
    
    political_score = detect_political_context(text)
    emotion_scores = get_emotion_scores(text)
    anger_score = emotion_scores.get("anger", 0.0)

    # Armazenar as pontuações em um dicionário
    scores = {
        'clickbait': clickbait_score,
        'hate_speech': hate_speech_score,
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
        "political_score": f"{political_score:.2f}",
        "anger_score": f"{anger_score:.2f}",
        "ragebait_classification": ragebait_classification
    })