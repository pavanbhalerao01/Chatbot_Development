import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import random


file_path = 'Conversation.csv'  
data = pd.read_csv(file_path)


input_texts = data['input_text'].tolist()
output_texts = data['output_text'].tolist()


output_texts = ['<START> ' + text + ' <END>' for text in output_texts]


input_tokenizer = Tokenizer()
output_tokenizer = Tokenizer()

input_tokenizer.fit_on_texts(input_texts)
output_tokenizer.fit_on_texts(output_texts)

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
output_sequences = output_tokenizer.texts_to_sequences(output_texts)

max_input_length = max([len(seq) for seq in input_sequences])
max_output_length = max([len(seq) for seq in output_sequences])

encoder_input = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
decoder_input = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')


decoder_output = np.zeros_like(decoder_input)
decoder_output[:, :-1] = decoder_input[:, 1:]


encoder_input_train, encoder_input_test, decoder_input_train, decoder_input_test, decoder_output_train, decoder_output_test = train_test_split(
    encoder_input, decoder_input, decoder_output, test_size=0.2, random_state=42
)


embedding_dim = 128
lstm_units = 256


encoder_inputs = tf.keras.Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]


decoder_inputs = tf.keras.Input(shape=(max_output_length,))
decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


batch_size = 64
epochs = 30

model.fit(
    [encoder_input_train, decoder_input_train],
    np.expand_dims(decoder_output_train, -1),
    validation_data=([encoder_input_test, decoder_input_test], np.expand_dims(decoder_output_test, -1)),
    batch_size=batch_size,
    epochs=epochs
)


encoder_model = tf.keras.Model(encoder_inputs, encoder_states)


model.save_weights("chatbot_model.weights.h5")
print("Model weights saved successfully.")



decoder_state_input_h = tf.keras.Input(shape=(lstm_units,))
decoder_state_input_c = tf.keras.Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_inf = decoder_embedding(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        
        for word, index in output_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_output_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return decoded_sentence.strip()


def chat():
    print("Chatbot: Hi! Type 'exit' to end the conversation.")
    while True:
        input_text = input("You: ")
        if input_text.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        input_seq = input_tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')
        
        response = decode_sequence(input_seq)
        print(f"Chatbot: {response}")


chat()
