import os
import base64
import logging
from flask import Flask, send_file, request, jsonify
from google import genai

# Configure logging for security and debugging in Google Cloud
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Fetch the API key securely
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    logger.error("CRITICAL: GEMINI_API_KEY environment variable is missing!")

@app.route('/')
def index():
    """Serve the main frontend application."""
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle real-time frame analysis during the session."""
    if not API_KEY:
        return jsonify({"error": "Server configuration error."}), 500

    try:
        data = request.get_json()
        
        # Security 1: Input Validation
        if not data or 'image' not in data:
            return jsonify({"error": "Invalid request: Missing image data."}), 400

        image_data = data.get('image')
        mode = data.get('mode', 'desk') 

        # Security 2: Strict mode validation
        valid_modes = ['desk', 'rehab', 'yoga', 'senior']
        if mode not in valid_modes:
            mode = 'desk'

        prompts = {
            'desk': "You are an AI posture coach for desk workers. Look at the image. Give ONE very short, encouraging sentence to correct their posture (e.g., shoulders, back).",
            'rehab': "You are a post-injury rehab specialist. Look at the image. Give ONE short, gentle instruction. Ensure they are moving safely.",
            'yoga': "You are a Yoga instructor. Look at the form in the image. Give ONE short cue about balance or breathing.",
            'senior': "You are a gentle physical therapist for seniors. Look at the image. Give ONE very short, encouraging, and safe instruction."
        }
        
        prompt = prompts.get(mode)

        # Security 3: Safe Base64 decoding
        try:
            if "," in image_data:
                base64_str = image_data.split(',')[1]
            else:
                base64_str = image_data
            image_bytes = base64.b64decode(base64_str)
        except Exception:
            return jsonify({"error": "Invalid image format."}), 400
        
        # Initialize Gemini Client and call the model
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, {'mime_type': 'image/jpeg', 'data': image_bytes}]
        )
        
        return jsonify({"feedback": response.text.strip()})
    
    except Exception as e:
        # Security 4: Error masking (Don't expose internal errors to the client)
        logger.error(f"Analysis Error: {str(e)}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500

@app.route('/report', methods=['POST'])
def report():
    """Generate the final session report."""
    if not API_KEY:
        return jsonify({"error": "Server configuration error."}), 500

    try:
        data = request.get_json()
        
        # Input Validation for report
        if not data or 'history' not in data:
            return jsonify({"error": "Invalid request: Missing session history."}), 400

        session_history = data.get('history', [])
        
        if not session_history:
            return jsonify({
                "score": "0/10", 
                "strengths": "No data collected.", 
                "improvements": "Try starting a new session and staying in the frame.", 
                "next_step": "Ensure your camera is working properly."
            })
            
        history_text = "\n".join(session_history)
        
        prompt = f"""
        Based on the following feedback given to a user during their physical therapy/fitness session:
        {history_text}
        
        Provide a final JSON report with these exact keys:
        - "score": A score out of 10 (e.g., "8/10").
        - "strengths": One short sentence on what they did well.
        - "improvements": One short sentence on what they need to work on.
        - "next_step": One short actionable advice for tomorrow.
        """
        
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        
        import json
        report_data = json.loads(response.text)
        
        # Security 5: Ensure Gemini returned all required keys before sending to frontend
        required_keys = ['score', 'strengths', 'improvements', 'next_step']
        for key in required_keys:
            if key not in report_data:
                report_data[key] = "Data unavailable"

        return jsonify(report_data)
        
    except Exception as e:
        logger.error(f"Report Error: {str(e)}")
        return jsonify({"error": "Failed to generate report."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    # Security 6: debug=False is CRITICAL for production safety
    app.run(debug=False, host='0.0.0.0', port=port)
