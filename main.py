import os, json, base64, asyncio, logging, uvicorn
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
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1beta'})
    
    try:
        # تعليمات باش Gemini يبدا الهضرة فالبلاصة غير يوقع الاتصال
        instr = "You are Kinetix AI. IMMEDIATELY start the session by greeting the user. Tell them you are ready to watch their movements. Be energetic!"
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"], 
            system_instruction=types.Content(parts=[types.Part.from_text(text=instr)])
        )
        
        # الموديل المستقر اللي بان فـ Logs ديالك
        model_id = "gemini-2.5-flash-native-audio-preview-12-2025"
        
        async with client.aio.live.connect(model=model_id, config=config) as session:
            logger.info(f"🟢 Session Active: {model_id}")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        if "image" in msg:
                            # صيفط الصورة لـ Gemini
                            await session.send(input=types.LiveClientContent(turns=[types.Content(parts=[types.Part.from_bytes(data=base64.b64decode(msg["image"].split(',')[1]), mime_type="image/jpeg")])]))
                        if "audio" in msg:
                            # صيفط الصوت لـ Gemini
                            await session.send(input=types.LiveClientContent(turns=[types.Content(parts=[types.Part.from_bytes(data=base64.b64decode(msg["audio"]), mime_type="audio/pcm;rate=16000")])]))
                except Exception: pass

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    await websocket.send_json({"audio": audio_b64})
                except Exception: pass

            await asyncio.gather(receive_from_client(), receive_from_gemini())
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
