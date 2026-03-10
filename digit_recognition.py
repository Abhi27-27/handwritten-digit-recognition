import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# -----------------------------
# 1. Show sample images
# -----------------------------
print("Sample images from MNIST dataset")

for i in range(5):
    plt.imshow(x_train[i], cmap='gray')
    plt.title("Label: " + str(y_train[i]))
    plt.axis("off")
    plt.show()

# -----------------------------
# 2. Data preprocessing
# -----------------------------

# Normalize pixel values (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images (28x28 → 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# -----------------------------
# 3. Build Neural Network Model
# -----------------------------
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Print model architecture
print("\nModel Architecture:")
model.summary()

# -----------------------------
# 4. Compile the model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 5. Train the model
# -----------------------------
print("\nTraining the model...")
model.fit(x_train, y_train, epochs=10, batch_size=32)

# -----------------------------
# 6. Evaluate model
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_acc)

# -----------------------------
# 7. Predictions
# -----------------------------
predictions = model.predict(x_test)

print("\nPredictions on test images")

for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(
        "Predicted: " + str(np.argmax(predictions[i])) +
        " | Actual: " + str(y_test[i])
    )
    plt.axis("off")
    plt.show()
