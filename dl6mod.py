import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

num_notes = 50
sequence_length = 30
num_sequences = 1000

np.random.seed(42)
data = np.random.randint(0, num_notes, size=(num_sequences, sequence_length))

x_train = data[:, :-1]
y_train = data[:, 1:]

x_train = to_categorical(x_train, num_classes=num_notes)
y_train = to_categorical(y_train, num_classes=num_notes)

input_layer = Input(shape=(sequence_length - 1, num_notes))
rnn1 = layers.Bidirectional(layers.SimpleRNN(128, return_sequences=True))(input_layer)
attention1 = layers.Attention()([rnn1, rnn1])
dropout1 = layers.Dropout(0.3)(attention1)
rnn2 = layers.Bidirectional(layers.SimpleRNN(128, return_sequences=True))(dropout1)
attention2 = layers.Attention()([rnn2, rnn2])
dropout2 = layers.Dropout(0.3)(attention2)
output_layer = layers.TimeDistributed(layers.Dense(num_notes, activation='softmax'))(dropout2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_training_history(history)

def generate_music(model, start_sequence, num_generated_notes=50):
    generated = []
    current_sequence = start_sequence
    for _ in range(num_generated_notes):
        pred = model.predict(current_sequence, verbose=0)
        next_note = np.argmax(pred[:, -1, :], axis=-1)
        generated.append(next_note[0])
        next_note_one_hot = to_categorical(next_note, num_classes=num_notes)
        current_sequence = np.concatenate((current_sequence[:, 1:, :], next_note_one_hot[:, np.newaxis, :]), axis=1)
    return generated

start_sequence = x_train[:1]
generated_music = generate_music(model, start_sequence, num_generated_notes=50)

def plot_piano_roll(generated_music, num_notes):
    piano_roll = np.zeros((num_notes, len(generated_music)))
    for t, note in enumerate(generated_music):
        piano_roll[note, t] = 1
    plt.figure(figsize=(15, 6))
    sns.heatmap(
        piano_roll,
        cmap="coolwarm",
        cbar=True,
        xticklabels=10,
        yticklabels=True,
        linewidths=0.1, linecolor='gray'
    )
    plt.title("Piano Roll Representation of Generated Music")
    plt.xlabel("Time Steps")
    plt.ylabel("Notes")
    plt.yticks(ticks=np.arange(0, num_notes, step=5), labels=np.arange(0, num_notes, step=5))
    plt.xticks(ticks=np.arange(0, len(generated_music), step=5), labels=np.arange(0, len(generated_music), step=5))
    plt.show()

print("Generated music sequence:", generated_music)
plot_piano_roll(generated_music, num_notes)
