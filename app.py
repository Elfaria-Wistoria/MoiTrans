from flask import Flask, render_template, request, jsonify
from translator import AdvancedMoiTranslator
import os

app = Flask(__name__)
translator = AdvancedMoiTranslator()

# Load atau train model saat startup
MODEL_PATH = 'models/moi_translator_model.joblib'
if os.path.exists(MODEL_PATH):
    translator.load_model(MODEL_PATH)
else:
    translator.train('Kamus MOI.xlsx')
    translator.save_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate')
def translate_page():
    return render_template('translate.html')

@app.route('/api/translate', methods=['POST'])
def translate():
    try:
        text = request.form.get('text', '')
        direction = request.form.get('direction', 'moi_to_indo')
        
        if not text:
            return jsonify({'error': 'Teks tidak boleh kosong'}), 400
            
        result = translator.translate(text, direction)
        return jsonify({'translation': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)