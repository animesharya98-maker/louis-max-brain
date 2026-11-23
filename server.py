import os
import io
import json
import asyncio
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import Groq
import edge_tts
from duckduckgo_search import DDGS
import yt_dlp

app = FastAPI()

# Apni free Groq API key yahan ya Render Environment Variables mein daal
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "APNI_GROQ_API_KEY_YAHAN_DAAL")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
Tu 'Louis Max' hai — Abhinandan (Arjun) ka personal AI voice assistant.
Tu smart, friendly, thoda witty, aur Hindi/Hinglish/English mein natural baat karta hai.
Rules:
1. Jawab hamesha chhota aur crisp rakh (2-3 lines max) taaki bolne mein fast lage.
2. Agar user koi gaana bajane ko bole (e.g., "Play Softly", "Karan Aujla ka gaana baja"), toh sirf JSON format mein reply kar:
   {"action": "play_music", "query": "<exact song and artist name>"}
3. Agar user live info, news, score, weather, ya current fact pooche, toh pehle search context ka use kar.
"""

def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {e}"

def get_youtube_audio_url(song_query: str):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'noplaylist': True,
        'default_search': 'ytsearch1'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(song_query, download=False)
        if 'entries' in info:
            info = info['entries'][0]
        return info['url'], info.get('title', song_query)

async def stream_pcm_to_esp32(websocket: WebSocket, audio_source: str):
    """Kisi bhi audio file ya YouTube URL ko 16kHz 16-bit Mono PCM mein convert karke ESP32 ko bhejta hai"""
    cmd = [
        "ffmpeg", "-i", audio_source,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "pipe:1"
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        while True:
            chunk = await process.stdout.read(2048)
            if not chunk:
                break
            await websocket.send_bytes(chunk)
            await asyncio.sleep(0.05) # Buffer overflow rokne ke liye pacing
    finally:
        if process.returncode is None:
            process.kill()

async def text_to_speech_stream(websocket: WebSocket, text: str):
    communicate = edge_tts.Communicate(text, "hi-IN-MadhurNeural") # Natural Indian Male Voice
    temp_mp3 = "/tmp/reply.mp3"
    await communicate.save(temp_mp3)
    await stream_pcm_to_esp32(websocket, temp_mp3)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Louis Max ESP32 Connected!")
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            # ESP32 se WAV audio bytes receive kar
            audio_bytes = await websocket.receive_bytes()
            if len(audio_bytes) < 3200:
                continue

            await websocket.send_text(json.dumps({"status": "Thinking..."}))

            # 1. Speech-to-Text (Groq Whisper)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "speech.wav"
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo"
            )
            user_text = transcription.text.strip()
            print(f"User: {user_text}")
            if not user_text:
                continue

            # 2. Check if web search is needed
            search_triggers = ["news", "score", "weather", "kaun hai", "kya hai", "latest", "price", "aaj", "search"]
            context_note = ""
            if any(word in user_text.lower() for word in search_triggers):
                await websocket.send_text(json.dumps({"status": "Searching..."}))
                web_data = search_web(user_text)
                context_note = f"\n[Live Web Search Results:\n{web_data}]"

            chat_history.append({"role": "user", "content": user_text + context_note})

            # 3. LLM Response (Groq Llama 3.3 70B)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=chat_history[-8:],
                temperature=0.7,
                max_tokens=200
            )
            reply = completion.choices[0].message.content.strip()
            chat_history.append({"role": "assistant", "content": reply})
            print(f"Louis Max: {reply}")

            # 4. Check if Music Action triggered
            if '"action": "play_music"' in reply:
                try:
                    data = json.loads(reply[reply.find("{"):reply.rfind("}")+1])
                    song_query = data["query"]
                    await websocket.send_text(json.dumps({"status": f"Playing: {song_query[:18]}"}))
                    await text_to_speech_stream(websocket, f"Chala raha hoon {song_query}")
                    stream_url, title = get_youtube_audio_url(song_query)
                    await websocket.send_text(json.dumps({"status": f"♪ {title[:18]}"}))
                    await stream_pcm_to_esp32(websocket, stream_url)
                except Exception as e:
                    await text_to_speech_stream(websocket, "Sorry bhai, ye gaana load nahi ho paya.")
            else:
                await websocket.send_text(json.dumps({"status": "Speaking..."}))
                await text_to_speech_stream(websocket, reply)

            await websocket.send_text(json.dumps({"status": "Listening..."}))

    except WebSocketDisconnect:
        print("ESP32 Disconnected")
