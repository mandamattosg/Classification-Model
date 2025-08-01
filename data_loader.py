import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, train_dir, val_dir, img_size, batch_size):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def load_data(self):
        """
        Configures and loads the training and validation datasets using ImageDataGenerator.
        """
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
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        val_gen = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False  # Recommended for validation data to maintain order
        )
        
        return train_gen, val_gen

