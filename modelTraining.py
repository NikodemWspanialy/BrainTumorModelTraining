import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Definicja parametrów
input_shape = (256, 256, 3)  # Rozmiar obrazów wejściowych
num_classes = 4  # Liczba kategorii: glioma, meningioma, notumor, pituitary
batch_size = 32
epochs = 10

# Ścieżki do danych
train_dir = 'cleaned\Training'
validation_dir = 'cleaned\Testing'

# Generowanie danych treningowych i walidacyjnych
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical')

# Tworzenie modelu CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size)

# Zapisanie modelu
model.save('model_guz_mozgu.h5')

# Tworzenie tabeli z wynikami treningu i walidacji
history_df = pd.DataFrame(history.history)
history_df.index += 1  # Indeksowanie epok od 1
print(history_df)