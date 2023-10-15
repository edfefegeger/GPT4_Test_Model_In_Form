# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_caching import Cache

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Форма для ввода текста</title>
    </head>
    <body>
        <h1>Форма для ввода текста</h1>
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



                // fetch('/generate', {
                //     method: 'POST',
                //     headers: {
                //         'Content-Type': 'application/json'
                //     },
                //     body: JSON.stringify({
                //         prompt: prompt,
                //         max_length: maxLength
                //     })
                // })
                // .then(response => response.json())
                // .then(data => {
                //     document.getElementById('generated-text').innerText = 'Сгенерированный текст: ' + data.generated_text;
                // })
                // .catch(error => console.error(error));

                // generated_text = process_text(prompt, maxLength);
                // document.getElementById('generated-text').innerText = 'Сгенерированный текст: ' + generated_text;
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
