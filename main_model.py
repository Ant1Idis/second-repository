from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
import tensorflow as tf

DATASET_PATH = 'datasets/'
IMG_SIZE = (128, 128)
BATCH = 32
SEED = 42

# Generators: fixed split, no shuffle on val
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen   = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", subset="training", shuffle=True, seed=SEED
)
val_data = val_gen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", subset="validation", shuffle=False, seed=SEED
)

num_classes = train_data.num_classes
print("Class indices:", train_data.class_indices)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    # If you add extra Dense layers, add a little Dropout to avoid collapse:
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.1),

    Dense(num_classes, activation="softmax")
])

# without label smoothing first to confirm it learns
loss_fn = "categorical_crossentropy"

# Lower LR a bit when the head is larger
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=loss_fn, metrics=["accuracy"])

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
)

test_loss, test_accuracy = model.evaluate(val_data, verbose=0)
print(f"Validation accuracy: {test_accuracy:.2f}")
model.save("image_classifier.h5")
