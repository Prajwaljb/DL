import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

num_words = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

x_train = pad_sequences(X_train, maxlen=maxlen)
x_test = pad_sequences(X_test, maxlen=maxlen)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=128, input_length=maxlen))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=64)

def predict_with_confidence(review):
    word_index = imdb.get_word_index()
    encoded_review = [word_index[word] + 3 for word in review.split() if word in word_index]
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)
    prediction = model.predict(padded_review, verbose=0)[0][0]
    if prediction > 0.7:
        category = "High Confidence"
    elif prediction > 0.4:
        category = "Moderate Confidence"
    else:
        category = "Low Confidence"
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, category, prediction

sample_reviews = [
    "This movie was fantastic with great acting and a solid plot",
    "I did not enjoy this movie at all. It was boring and predictable",
    "An average movie with some good moments but overall forgettable",
]

for review in sample_reviews:
    sentiment, confidence_level, score = predict_with_confidence(review)
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence Level: {confidence_level} (Score: {score:.2f})")
    print("-" * 50)
