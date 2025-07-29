import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


dataset_path = 'dataset'
img_size = (48, 48)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    class_mode='categorical',
    subset='validation'
)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)

model.save('mood_model.h5')
print("✅ Model trained and saved as mood_model.h5")
