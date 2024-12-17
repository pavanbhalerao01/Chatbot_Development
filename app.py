from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


file_path = 'Conversation.csv'
data = pd.read_csv(file_path)


input_texts = data['input_text'].tolist()
output_texts = ['<START> ' + text + ' <END>' for text in data['output_text'].tolist()]


input_tokenizer = Tokenizer()
output_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
output_tokenizer.fit_on_texts(output_texts)

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1


max_input_length = max(len(s.split()) for s in input_texts)
max_output_length = max(len(s.split()) + 2 for s in output_texts)  # Account for START/END tokens


embedding_dim = 128
lstm_units = 256

# Encoder Model
encoder_inputs = tf.keras.Input(shape=(max_input_length,))
encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(lstm_units, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder Model
decoder_inputs = tf.keras.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')
output = decoder_dense(decoder_outputs)


model = tf.keras.Model([encoder_inputs, decoder_inputs], output)

model.load_weights("chatbot_model.weights.h5")
 

# Inference Setup
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.Input(shape=(lstm_units,))
decoder_state_input_c = tf.keras.Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_inf = decoder_embedding(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedding_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states_inf
)


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        for word, index in output_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word == "<end>" or len(decoded_sentence.split()) > max_output_length:
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    input_seq = input_tokenizer.texts_to_sequences([user_message])
    input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')

    response = decode_sequence(input_seq)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
