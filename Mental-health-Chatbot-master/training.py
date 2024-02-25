import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open('intents.json') as file:
    data = json.load(file)

# Extracting patterns and intents
patterns = []
intents = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents.append(intent['tag'])

# Tokenize the patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1

# Convert text data to sequences
sequences = tokenizer.texts_to_sequences(patterns)
max_sequence_len = max([len(x) for x in sequences])

# Pad sequences to ensure uniform input size
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# Convert intents to numerical labels
label_encoder = LabelEncoder()
label_encoder.fit(intents)
encoded_intents = label_encoder.transform(intents)

# Define the model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_sequence_len),
    Bidirectional(LSTM(128)),
    Dense(64, activation='relu'),
    Dense(len(set(encoded_intents)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, encoded_intents, epochs=100, verbose=1)

# Save the model
model.save('my_model.keras')