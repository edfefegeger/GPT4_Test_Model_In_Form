from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer
import torch

app = Flask(__name__)
TOKEN = "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"

pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf", torch_dtype=torch.float32, device=0, token=TOKEN)  # Используйте новую модель
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GPT-3 Alpaca Generator</title>
    </head>
    <body>
        <h1>GPT-3 Alpaca Generator</h1>
        <form id="generate-form">
            <label for="prompt">Введите текст:</label>
            <input type="text" id="prompt" name="prompt" required>
            <br>
            <button type="submit">Сгенерировать</button>
        </form>
        <div id="generated-text"></div>

        <script>
            document.getElementById('generate-form').addEventListener('submit', function(event) {
                event.preventDefault();
                const prompt = document.getElementById('prompt').value;

                fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: prompt
                    })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('generated-text').innerText = 'Сгенерированный текст: ' + data.generated_text;
                })
                .catch(error => console.error(error));
            });
        </script>
    </body>
    </html>
    '''

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data['prompt']

        # Добавьте специальные токены и обработайте текст с использованием новой модели
        input_text = f"You are a friendly chatbot who always responds in the style of a pirate. User: {prompt}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        outputs = pipe(input_ids.to('cuda')).to('cpu')
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
