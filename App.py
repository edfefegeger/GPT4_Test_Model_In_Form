from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

model_name = "HuggingFaceH4/zephyr-7b-alpha"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

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

@app.route('/', methods=['GET'])
def generate_text():
    try:
        prompt = request.args.get('text', '')
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        
        generated_text = ' '.join(dict.fromkeys(generated_text.split()))

        return jsonify({'generated_text': generated_text})
    except Exception as e:
        print(f"Error: {str(e)}")

        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
