from flask import Flask, request, jsonify
import openai
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

openai.api_key = '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'

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
                        max_length: parseInt(maxLength)  // Преобразовать в целое число
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
@cache.cached(timeout=60)  # кэширование на 60 секунд
def generate_text():
    try:
        data = request.get_json()
        prompt = data['prompt']
        max_length = data.get('max_length', 100)

        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=max_length,
            n=1
        )

        generated_text = response.choices[0].text.strip()
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        print(f"Error: {str(e)}")
        # Запись информации об ошибке в лог-файл или консоль
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
