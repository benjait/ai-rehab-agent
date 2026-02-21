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
        instr = (
            "You are Kinetix AI, an energetic real-time fitness coach. "
            "CRITICAL RULE: Always detect the language the user speaks and respond in that EXACT language. "
            "Arabic (Darija/MSA), French, English, Spanish — any language is fine. "
            "Greet the user immediately when the session starts. "
            "Listen carefully to everything they say and respond to their questions and comments. "
            "Also watch their camera movements and give short, motivating coaching feedback."
        )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part.from_text(text=instr)])
        )

        model_id = "gemini-2.5-flash-native-audio-preview-12-2025"

        async with client.aio.live.connect(model=model_id, config=config) as session:
            logger.info(f"🟢 Session Active: {model_id}")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)

                        # ✅ CRITICAL: Send audio and image as SEPARATE calls
                        # Mixing them in one media_chunks breaks VAD (voice detection)

                        if "audio" in msg:
                            audio_bytes = base64.b64decode(msg["audio"])
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")]
                                )
                            )
                            logger.debug(f"🎤 Audio: {len(audio_bytes)} bytes")

                        if "image" in msg:
                            image_bytes = base64.b64decode(msg["image"].split(',')[1])
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[types.Blob(data=image_bytes, mime_type="image/jpeg")]
                                )
                            )
                            logger.debug("📷 Image frame sent")

                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"receive_from_client error: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    await websocket.send_json({"audio": audio_b64})
                                    logger.info("🔊 Audio sent to client")

                except Exception as e:
                    logger.error(f"receive_from_gemini error: {e}")

            await asyncio.gather(receive_from_client(), receive_from_gemini())

    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
