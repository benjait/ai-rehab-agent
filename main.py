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
        await websocket.send_json({"error": "Missing GEMINI_API_KEY in Google Cloud"})
        await websocket.close()
        return

    try:
        client = genai.Client(api_key=API_KEY)
        
        config = types.LiveConnectConfig(
            response_modalities=[types.LiveModality.AUDIO],
            system_instruction=types.Content(parts=[
                types.Part.from_text("You are an expert physical therapy and fitness AI coach. You are watching a live video feed of the user. Guide them, correct their posture, and encourage them in real-time. Keep your instructions extremely short, energetic, and natural.")
            ])
        )
        
        # الاتصال مع Gemini
        async with client.aio.live.connect(model="gemini-2.0-flash-exp", config=config) as session:
            logger.info("🟢 Connected to Gemini Live API")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        
                        # تبسيط الإرسال باش نتفاداو الأخطاء ديال النسخ القديمة د المكتبة
                        if "image" in msg:
                            image_b64 = msg["image"].split(',')[1] if "," in msg["image"] else msg["image"]
                            image_bytes = base64.b64decode(image_b64)
                            await session.send(input={"mime_type": "image/jpeg", "data": image_bytes})
                            
                        if "audio" in msg:
                            audio_bytes = base64.b64decode(msg["audio"])
                            await session.send(input={"mime_type": "audio/pcm;rate=16000", "data": audio_bytes})
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    logger.error(f"Client error: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    await websocket.send_json({"audio": audio_b64})
                except Exception as e:
                    logger.error(f"Gemini error: {e}")

            await asyncio.gather(receive_from_client(), receive_from_gemini())

    except Exception as e:
        # 🔥 هادا هو السر: أي مشكل غيوقع غيصيفطو ليك للتليفون تقراه بعينيك!
        error_msg = f"API Error: {str(e)}"
        logger.error(error_msg)
        try:
            await websocket.send_json({"error": error_msg})
        except:
            pass
        await websocket.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
