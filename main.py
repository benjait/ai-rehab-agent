import os
import base64
import logging
from flask import Flask, send_file, request, jsonify
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not API_KEY:
        return jsonify({"error": "Server configuration error."}), 500

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data."}), 400

        image_data = data.get('image')
        mode = data.get('mode', 'desk') 
        # جبنا الذاكرة باش الوكيل يعقل أشنو قال للمستخدم
        history = data.get('history', [])

        valid_modes = ['desk', 'rehab', 'yoga', 'senior']
        if mode not in valid_modes:
            mode = 'desk'

        # هنا بدلنا العقلية ديال الذكاء الاصطناعي باش يولي هو "القائد"
        history_text = "Session just started." if not history else "\n".join(history[-3:]) # كنعطيوه غير اخر 3 جمل باش مايتلفش
        
        prompt = f"""
        You are a proactive AI Coach leading a '{mode}' physical session. 
        You MUST LEAD the session. Do not just passively observe.
        
        Here is what you have instructed the user so far:
        [{history_text}]
        
        YOUR TASK:
        1. If the session just started (no history), warmly greet the user and INSTRUCT them to do their FIRST specific exercise.
        2. If an exercise is ongoing, look at the user's image. Are they doing the exercise you asked for? If wrong, correct their form gently.
        3. If they are doing it perfectly, praise them and INSTRUCT them to move to the NEXT exercise.
        
        RULES:
        - Keep it very conversational, energetic, and natural.
        - NEVER give long paragraphs. ONLY 1 or 2 short sentences.
        - Talk directly to the user (e.g., "Great job! Now, let's stretch your arms...").
        """

        try:
            base64_str = image_data.split(',')[1] if "," in image_data else image_data
            image_bytes = base64.b64decode(base64_str)
        except Exception:
            return jsonify({"error": "Invalid image format."}), 400
        
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, {'mime_type': 'image/jpeg', 'data': image_bytes}]
        )
        
        return jsonify({"feedback": response.text.strip()})
    
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/report', methods=['POST'])
def report():
    # ... (كود التقرير بقى كيفما هو ماتقيسوش، راه ناضي) ...
    if not API_KEY:
        return jsonify({"error": "Server configuration error."}), 500

    try:
        data = request.get_json()
        session_history = data.get('history', [])
        
        if not session_history:
            return jsonify({
                "score": "0/10", 
                "strengths": "No data collected.", 
                "improvements": "Try starting a new session.", 
                "next_step": "Ensure your camera is working."
            })
            
        history_text = "\n".join(session_history)
        prompt = f"""
        Based on this session history: {history_text}
        Provide a JSON report with exact keys: "score" (out of 10), "strengths", "improvements", "next_step". Keep values short.
        """
        
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        import json
        report_data = json.loads(response.text)
        return jsonify(report_data)
        
    except Exception as e:
        logger.error(f"Report Error: {str(e)}")
        return jsonify({"error": "Failed to generate report."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
