import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import os

# Define dataset path
BASE_DIR = r"D:\DR Detection-20241209T040612Z-001\DR Detection\2217041\gaussian_filtered_images\gaussian_filtered_images"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

# Prepare training and validation datasets
train_data = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(256, 256),
    batch_size=32,
    class_mode="sparse",
    subset="training"
)

validation_data = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(256, 256),
    batch_size=32,
    class_mode="sparse",
    subset="validation"
)

# Learning rate schedule function
# Learning rate schedule function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))
 

# Build the improved CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    GlobalAveragePooling2D(),

    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# Compile the model with advanced optimizer
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    LearningRateScheduler(scheduler, verbose=1)
]

# Train the model
history = model.fit(
    train_data,
    epochs=40,
    validation_data=validation_data,
    callbacks=callbacks
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_data)
print(f"Validation Accuracy: {val_accuracy:.2f}")
