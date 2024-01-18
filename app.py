import random
import csv
from flask import Flask, render_template
from flask_socketio import SocketIO
from datetime import datetime
import socket
from keras.preprocessing import text, sequence
from process import denoise_text
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

socketio = SocketIO(app, cors_allowed_origins="*")
model = load_model('model/model_dir.h5')
tokenizer = joblib.load('model/tokenizer_joblib.joblib')

# with open('model/87-80800104141235model.pkl', 'rb') as file:
#     model = pickle.load(file)
#
#
# with open('model/tokenizer_87-80800104141235.pk', 'rb') as file:
#     tokenizer = pickle.load(file)


def load_messages():
    try:
        with open('messages.csv', 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            messages = list(reader)
    except FileNotFoundError:
        messages = []
    return messages

messages = load_messages()

def save_messages():
    with open('messages.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['id', 'msg', 'sentiment', 'probability', 'timestamp']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(messages)

def get_sentiment_probability(inp_text):
    inp_text = denoise_text(inp_text)
    tokenized_test = tokenizer.texts_to_sequences([inp_text])
    inp_text = sequence.pad_sequences(tokenized_test, maxlen=300)
    pred = model.predict(inp_text)
    sentiments = None
    if(pred>=0.5):
        sentiments="Positive"
    else:
        sentiments = "Negative"
    print(sentiments,"-----",pred)
    return {
        'sentiment': sentiments,
        'probability': round(pred[0][0], 2)
    }

@app.route('/')
def receive():
    global messages
    messages = load_messages()
    return render_template('receive.html', messages=messages)

@app.route('/send')
def send():
    return render_template('send.html')

@socketio.on('text', namespace='/chat')
def text(message):
    global messages
    if message['msg'].strip():
        sentiment_probability = get_sentiment_probability(message['msg'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_message = {
            'id': len(messages) + 1,
            'msg': message['msg'],
            'sentiment': sentiment_probability['sentiment'],
            'probability': (sentiment_probability['probability']*100),
            'timestamp': timestamp
        }
        messages.append(new_message)
        socketio.emit('message', new_message, namespace='/chat')
        save_messages()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=0, allow_unsafe_werkzeug=True)

