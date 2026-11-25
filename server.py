import os
import json
import asyncio
import tempfile
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq
import edge_tts
from duckduckgo_search import DDGS
import yt_dlp

app = FastAPI()
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/")
async def root():
    return {"status": "Louis Max Brain Online", "ws_endpoint": "/ws"}

def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No search results found."
            return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"

def get_song_pcm(song_query: str) -> bytes:
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "default_search": "ytsearch1",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(song_query, download=False)
            if "entries" in info:
                info = info["entries"][0]
            audio_url = info["url"]

        cmd = [
            "ffmpeg", "-i", audio_url,
            "-t", "20",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "pipe:1"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return proc.stdout
    except Exception as e:
        print("Music error:", e)
        return b""

async def text_to_pcm(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural")
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]

    cmd = [
        "ffmpeg", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=mp3_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout

async def stream_pcm_safely(ws: WebSocket, pcm_data: bytes):
    # Send small 1024-byte chunks so ESP32 SSL WebSocket never overflows or disconnects
    for i in range(0, len(pcm_data), 1024):
        await ws.send_bytes(pcm_data[i:i+1024])
        await asyncio.sleep(0.03)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("ESP32 Louis Max Connected!")
    await ws.send_text(json.dumps({"status": "Online! Listening..."}))

    try:
        while True:
            wav_bytes = await ws.receive_bytes()
            await ws.send_text(json.dumps({"status": "Thinking..."}))

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file
                )
            os.remove(tmp_path)
            user_text = transcript.text.strip()
            print("User said:", user_text)

            if not user_text:
                await ws.send_text(json.dumps({"status": "Online! Listening..."}))
                continue

            lower_text = user_text.lower()
            if "volume" in lower_text:
                vol = 1.0 if ("full" in lower_text or "max" in lower_text or "100" in lower_text) else (0.3 if "low" in lower_text else 0.7)
                await ws.send_text(json.dumps({"volume": vol}))
                pcm = await text_to_pcm(f"Volume set to {int(vol * 100)} percent.")
                await stream_pcm_safely(ws, pcm)
                await ws.send_text(json.dumps({"status": "Online! Listening..."}))
                continue

            if any(k in lower_text for k in ["play ", "song", "gaana", "music"]):
                await ws.send_text(json.dumps({"status": "Loading Song..."}))
                pcm_song = await asyncio.to_thread(get_song_pcm, user_text)
                if pcm_song:
                    await ws.send_text(json.dumps({"status": "Playing Music..."}))
                    await stream_pcm_safely(ws, pcm_song)
                else:
                    pcm = await text_to_pcm("Sorry, I could not load that song right now.")
                    await stream_pcm_safely(ws, pcm)
                await ws.send_text(json.dumps({"status": "Online! Listening..."}))
                continue

            context = ""
            if any(k in lower_text for k in ["search", "news", "weather", "latest", "who is", "what is", "score", "price", "today", "aaj", "kya hai"]):
                await ws.send_text(json.dumps({"status": "Searching Web..."}))
                context = await asyncio.to_thread(search_web, user_text)

            completion = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are Louis Max, a smart voice AI assistant built on an ESP32 by Abhinandan and Arjun. Keep answers crisp and clear in 2 short sentences for spoken output."},
                    {"role": "user", "content": f"User query: {user_text}\nWeb context: {context}"}
                ]
            )
            reply_text = completion.choices[0].message.content
            print("Louis Max:", reply_text)

            await ws.send_text(json.dumps({"status": "Speaking..."}))
            pcm_reply = await text_to_pcm(reply_text)
            await stream_pcm_safely(ws, pcm_reply)

            await ws.send_text(json.dumps({"status": "Online! Listening..."}))

    except WebSocketDisconnect:
        print("ESP32 disconnected")
