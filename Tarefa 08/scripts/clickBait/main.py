from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
from app.utils import (
    extract_video_info,
    extract_comments,
    analyze_comments,
    analyze_temporal,
    detect_political_context,
    compute_ragebait_score
)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Carregar modelos
clickbait_model = joblib.load('app/models/clickbait_model.pkl')
hatebert_model = joblib.load('app/models/hatebert_model.pkl')  # Carregue conforme necessário

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_video(request: Request, video_url: str = Form(...)):
    # Extrair informações do vídeo
    video_info = extract_video_info(video_url)
    title = video_info['title']
    description = video_info['description']
    publish_time = video_info['publish_time']

    # Classificar título e descrição
    text = f"{title} {description}"
    clickbait_score = clickbait_model.predict_proba([text])[0][1]

    # Extrair comentários
    comments = extract_comments(video_url, max_comments=10000)

    # Analisar comentários
    sentiment_score, hate_speech_score = analyze_comments(comments, hatebert_model)

    # Análise temporal
    temporal_score = analyze_temporal(comments, publish_time)

    # Detectar contexto político
    political_score = detect_political_context(title + ' ' + description)

    # Calcular pontuação final de Rage Bait
    ragebait_score = compute_ragebait_score(
        clickbait_score,
        sentiment_score,
        hate_speech_score,
        temporal_score,
        political_score
    )

    return templates.TemplateResponse("index.html", {
        "request": request,
        "video_url": video_url,
        "clickbait_score": f"{clickbait_score:.2f}",
        "sentiment_score": f"{sentiment_score:.2f}",
        "hate_speech_score": f"{hate_speech_score:.2f}",
        "temporal_score": f"{temporal_score:.2f}",
        "political_score": f"{political_score:.2f}",
        "ragebait_score": f"{ragebait_score:.2f}"
    })
