import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_data():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(10,)),
        layers.Dropout(0.3),
        layers.Dense(20, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    return model

def train_model_with_history(model, optimizer, X_train, y_train, X_val, y_val, batch_size, epochs, optimizer_name):
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    def scheduler(epoch, lr):
        if epoch % 10 == 0 and epoch != 0:
            return lr * 0.5
        return lr
    lr_scheduler = callbacks.LearningRateScheduler(scheduler)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=[lr_scheduler, early_stopping]
    )
    
    print(f"\nTraining with {optimizer_name}:")
    for epoch in range(len(history.history['loss'])):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        print(f"Epoch {epoch+1}/{epochs} - {optimizer_name} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return history.history['loss'], history.history['val_loss']

X_train, X_val, y_train, y_val = create_data()

model_sgd = create_model()
model_adam = create_model()

optimizer_sgd = optimizers.SGD(learning_rate=0.01)
optimizer_adam = optimizers.Adam(learning_rate=0.001)

epochs = 50
batch_size = 32

sgd_train_loss, sgd_val_loss = train_model_with_history(
    model_sgd, optimizer_sgd, X_train, y_train, X_val, y_val, batch_size, epochs, 'SGD'
)

adam_train_loss, adam_val_loss = train_model_with_history(
    model_adam, optimizer_adam, X_train, y_train, X_val, y_val, batch_size, epochs, 'Adam'
)

plt.plot(range(1, len(sgd_train_loss) + 1), sgd_train_loss, label='SGD - Train', color='blue')
plt.plot(range(1, len(sgd_val_loss) + 1), sgd_val_loss, label='SGD - Val', color='blue', linestyle='dashed')
plt.plot(range(1, len(adam_train_loss) + 1), adam_train_loss, label='Adam - Train', color='orange')
plt.plot(range(1, len(adam_val_loss) + 1), adam_val_loss, label='Adam - Val', color='orange', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Comparison Curves for SGD and Adam with Validation')
plt.grid(True)
plt.show()
