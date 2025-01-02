# !pip install transformers sentencepiece --quiet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# Проверяем наличие GPU, но не выводим ошибку, если его нет
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ['non-toxic', 'insult', 'obscenity', 'threat', 'dangerous']

def predict_toxicity_label(text):
    """Предсказывает метку токсичности и score для текста."""
    try:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        predicted_index = np.argmax(probabilities)
        predicted_label = labels[predicted_index]
        predicted_score = probabilities[predicted_index]  # Score - это максимальная вероятность

        return predicted_label, predicted_score

    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return "Ошибка", 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_label = None
    predicted_score = None
    input_text = None  # Инициализируем переменную для введенного текста

    if request.method == "POST":
        input_text = request.form["text"]  # Сохраняем введенный текст
        predicted_label, predicted_score = predict_toxicity_label(input_text)
    
    return render_template("index.html", predicted_label=predicted_label, predicted_score=predicted_score, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)

# Пример использования:
# texts = [
#     "Вань, ты дурак совсем?",
#     "Пусть оружие уступит тоге.",
#     "somtimes to stay alive you gotta kill your mind"
#     "Ты можешь выбрать любое животное, которое ты захочешь!",
#     "Идущие на смерть приветствуют тебя!",
#     "Или побеждать, или умирать!"
# ]

# for text in texts:
#     label, score = predict_toxicity_label(text)
#     print(f"Текст: '{text}', Метка: {label}, Score: {score:.4f}")