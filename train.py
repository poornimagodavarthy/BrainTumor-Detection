from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def generate_data(train_dir, test_dir):

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # data augmentation parameters
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=110,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
        channel_shift_range=0.2,
        vertical_flip=True,
        brightness_range=[0.5, 1.5]
    )

    # Validation data generator (no augmentation)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # generator for training data
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(512, 512),
        batch_size=64,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    # generator for validation data
    valid_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(512, 512),
        batch_size=64,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    # generator for testing data
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(512, 512),
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, test_generator, valid_generator
    


## MODEL TRAINING

def train_model(model_path, train_generator, test_generator, valid_generator):
    model = load_model(model_path)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.000066),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # Get class weights
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes )

    class_weights = dict(enumerate(class_weights))
    #class_weights[1] *=2.7
    #class_weights[0]*=1.5

    #print("Training Class:", train_generator.n)
    #print("Validation Class:", valid_generator.n)
    #print("Testing Class:", test_generator.n)


    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=1,
        validation_data=valid_generator,
        validation_steps=50,
        class_weight=class_weights,
    )

    print("Training complete.") 

    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = valid_generator.n // valid_generator.batch_size
    test_steps = test_generator.n // test_generator.batch_size



# Load the trained model
model_path = '/path/to/trained/model'

train_dir = '/path/to/resized/training/images'
test_dir = '/path/to/resized/testing/images'

train_generator, valid_generator, test_generator = generate_data(train_dir, test_dir)

model, history = train_model(model_path, train_generator, test_generator, valid_generator)

model.save('path/to/save/model')