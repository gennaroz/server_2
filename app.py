from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import datetime

app = Flask(__name__)
model = whisper.load_model("base")  # puoi usare "tiny", "base", "small"

@app.route('/speech', methods=['POST'])
def speech_to_text():
    try:
        # salva stream in file temporaneo
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(request.data)
        temp_file.close()

        print(f"[{datetime.datetime.now()}] Ricevuto file: {temp_file.name}")

        # trascrivi
        result = model.transcribe(temp_file.name, language='it')
        print("Testo:", result["text"])

        os.remove(temp_file.name)
        return jsonify({"text": result["text"]})

    except Exception as e:
        print("Errore:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
