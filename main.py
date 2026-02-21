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

    # التعديل هنا: استهداف Gemini 2.5 Flash
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})
    
    try:
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"], 
            system_instruction=types.Content(parts=[
                types.Part.from_text("You are Kinetix AI, a professional physical therapy coach. You are watching the user via a live feed. Give immediate, short, and clear voice instructions to guide their workout or rehab.")
            ])
        )
        
        # استبدال الموديل بـ gemini-2.5-flash
        async with client.aio.live.connect(model="gemini-2.5-flash", config=config) as session:
            logger.info("🟢 Connected to Gemini 2.5 Flash Live API")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        if "image" in msg:
                            image_b64 = msg["image"].split(',')[1] if "," in msg["image"] else msg["image"]
                            image_bytes = base64.b64decode(image_b64)
                            await session.send(input=types.LiveClientContent(
                                turns=[types.Content(parts=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")])]
                            ))
                        if "audio" in msg:
                            audio_bytes = base64.b64decode(msg["audio"])
                            await session.send(input=types.LiveClientContent(
                                turns=[types.Content(parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/pcm;rate=16000")])]
                            ))
                except Exception as e:
                    logger.error(f"Client stream error: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    await websocket.send_json({"audio": audio_b64})
                except Exception as e:
                    logger.error(f"Gemini 2.5 response error: {e}")

            await asyncio.gather(receive_from_client(), receive_from_gemini())

    except Exception as e:
        logger.error(f"Live Session Error: {e}")
        await websocket.send_json({"error": str(e)})
        await websocket.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
