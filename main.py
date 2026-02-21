import os
from flask import Flask, send_file, request, jsonify
from google import genai

app = Flask(__name__)

# هادا الساروت ديالك لي غنحطوه فـ Google Cloud من بعد
API_KEY = os.environ.get("GEMINI_API_KEY")

@app.route('/')
def index():
    # كيعرض الواجهة ديال التطبيق للمستخدم
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # هادي هي البلاصة لي كتستقبل الفيديو والصوت وكتصيفطهم لـ Gemini
    # فالمسابقة غنخدمو بـ Gemini 2.5 Flash لي كيدعم التفاعل المباشر
    try:
        client = genai.Client(api_key=API_KEY)
        
        # التعليمة السحرية باش يولي كوتش
        system_instruction = """أنت كوتش خبير في العلاج الطبيعي والرياضة. 
        حلل حركات المستخدم من الصور والصوت، وقدم نصائح قصيرة ومباشرة ومحفزة. 
        إذا كانت الحركة خاطئة صححها بلطف، وإذا كانت صحيحة شجعه."""
        
        # (هنا كيكون الكود ديال الربط المباشر مع الفيديو والصوت)
        # للتجربة المبدئية كنردّو جواب تجريبي
        return jsonify({"feedback": "كوتش Gemini: راك غادي مزيان، زيد هبط ظهرك شوية!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # باش يخدم فـ Google Cloud
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
