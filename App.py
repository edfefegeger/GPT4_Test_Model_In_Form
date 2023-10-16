from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

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

                fetch(`/?text=${encodeURIComponent(prompt)}`)
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

@app.route('/', methods=['POST'])
def generate_text():
    try:
        prompt = request.args.get('text', '')
        
        messages = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
            {"role": "user", "content": prompt},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]
        
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
