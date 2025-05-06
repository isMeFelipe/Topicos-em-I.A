from googleapiclient.discovery import build
import pandas as pd
import re
import os

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(youtube, video_id, video_url, max_comments=10000):
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100, # Limite máximo por requisição
                textFormat="plainText",
                pageToken=next_page_token
            ).execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "video_url": video_url,
                    "author": comment.get("authorDisplayName"),
                    "text": comment.get("textDisplay"),
                    "likes": comment.get("likeCount"),
                    "published_at": comment.get("publishedAt")
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    except Exception as e: # Quando o vídeo não tem os comentários habilitados ou a API falha
        print(f"[!] Erro ao buscar comentários para o vídeo {video_id}: {e}")
        return []

    return comments


def save_comments_to_csv(new_comments, filename="comentarios.csv"):
    new_df = pd.DataFrame(new_comments)

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["video_id", "text"], inplace=True)
    else:
        combined_df = new_df

    combined_df.to_csv(filename, index=False, encoding='utf-8-sig')

def main(url, api_key, filename="comentarios.csv"):
    video_id = extract_video_id(url)
    if not video_id:
        print("URL inválida ou vídeo não encontrado.")
        return

    # Verifica se já existe e se esse vídeo já foi baixado
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        if video_id in existing_df["video_id"].values:
            print(f"Vídeo {video_id} já foi processado. Ignorando.")
            return

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = get_comments(youtube, video_id, url)
    save_comments_to_csv(comments, filename)
    print(f"{len(comments)} comentários adicionados ao '{filename}'.")

main("https://www.youtube.com/watch?v=QhjkjXVAHiA&ab_channel=UOL", "SUA_API_KEY")
