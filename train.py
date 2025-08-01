import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import os

# Configurations 
EPOCHS = 50  
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_dir = r"horse-or-human"
val_dir = r"validation-horse-or-human"

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the datasets
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# MobileNetV2 with Transfer Learning 
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze the base model
base_model.trainable = False

# New head for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)  
predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# EarlyStopping - stops training when val_loss has stopped improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Save the model
model.save("model/person_vs_horse_model.h5")

# Save metrics
os.makedirs("results", exist_ok=True)

plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss during training')
plt.savefig('metrics_results/loss_plot.png')
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Acuracy during training')
plt.savefig('metrics_results/accuracy_plot.png')
plt.close()

val_gen.reset() # This is important to ensure the order of predictions matches the labels

val_datagen_eval = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_gen_eval = val_datagen_eval.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Important
)

# Predict on validation set
preds_eval = model.predict(val_gen_eval)
preds_binary = (preds_eval > 0.5).astype(int).flatten()
true_labels = val_gen_eval.classes

# Confusion matrix 
cm = confusion_matrix(true_labels, preds_binary)
print("\nConfusion matrix:\n", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Horse', 'Person'], yticklabels=['Horse', 'Person'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('metrics_results/confusion_matrix.png')
plt.close()

# Report
report = classification_report(true_labels, preds_binary, target_names=['Horse', 'Person'])
print("\nClassification report:\n", report)


