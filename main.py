import os
import json
import base64
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
API_KEY = os.environ.get("GEMINI_API_KEY")

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not API_KEY:
        await websocket.send_json({"error": "Missing API KEY"})
        await websocket.close()
        return

    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1beta'})
    
    try:
        # تعليمات صريحة لـ Gemini باش هو يبدا الهضرة (Initiate conversation)
        instr = "You are Kinetix AI, a professional coach. As soon as the session starts, GREET the user and ask them to show you their movements. Speak directly and be energetic."
        
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"], 
            system_instruction=types.Content(parts=[types.Part.from_text(text=instr)])
        )
        
        # الموديل لي خدام ليك فـ europe-west1 حسب اللوݣز ديالك
        model_id = "gemini-2.5-flash-native-audio-preview-12-2025"
        
        async with client.aio.live.connect(model=model_id, config=config) as session:
            logger.info(f"🟢 Active Session with {model_id}")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        
                        # معالجة الصور بحذر
                        if "image" in msg and msg["image"]:
                            try:
                                img_data = msg["image"].split(',')[1] if "," in msg["image"] else msg["image"]
                                await session.send(input=types.LiveClientContent(
                                    turns=[types.Content(parts=[types.Part.from_bytes(data=base64.b64decode(img_data), mime_type="image/jpeg")])]
                                ))
                            except: pass # تجاهل أخطاء الصور البسيطة

                        # معالجة الصوت
                        if "audio" in msg and msg["audio"]:
                            try:
                                await session.send(input=types.LiveClientContent(
                                    turns=[types.Content(parts=[types.Part.from_bytes(data=base64.b64decode(msg["audio"]), mime_type="audio/pcm;rate=16000")])]
                                ))
                            except: pass
                except Exception as e:
                    logger.error(f"Client Loop Error: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    await websocket.send_json({"audio": audio_b64})
                except Exception as e:
                    logger.error(f"Gemini Loop Error: {e}")

            await asyncio.gather(receive_from_client(), receive_from_gemini())

    except Exception as e:
        logger.error(f"Global Session Error: {e}")
        await websocket.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
