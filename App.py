# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_name = "chavinlo/gpt4-x-alpaca"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GPT-4 Alpaca Generator</title>
    </head>
    <body>
        <h1>GPT-4 Alpaca Generator</h1>
        <form id="generate-form">
            <label for="prompt">Введите текст:</label>
            <input type="text" id="prompt" name="prompt" required>
            <br>
            <label for="max-length">Максимальная длина текста (по умолчанию 100):</label>
            <input type="number" id="max-length" name="max_length" value="100">
            <br>
            <button type="submit">Сгенерировать</button>
        </form>
        <div id="generated-text"></div>

        <script>
            document.getElementById('generate-form').addEventListener('submit', function(event) {
                event.preventDefault();
                const prompt = document.getElementById('prompt').value;
                const maxLength = document.getElementById('max-length').value;

                fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: maxLength
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
    data = request.get_json()
    prompt = data['prompt']
    max_length = data.get('max_length', 100)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
