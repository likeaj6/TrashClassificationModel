import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import shutil

import numpy as np


IMAGE_WIDTH, IMAGE_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 50
BAT_SIZE = 50
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
ALL_DATA_FILEPATH = './data/dataset-original/'
TRAINING_DATA_FILEPATH = './data/training/'
VALIDATION_DATA_FILEPATH = './data/validation/'
VALIDATION_SPLIT = 0.25

def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(TRAINING_DATA_FILEPATH)
    nb_classes = len(glob.glob(TRAINING_DATA_FILEPATH + "/*"))
    nb_val_samples = get_nb_files(VALIDATION_DATA_FILEPATH)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # split_dataset_into_test_and_train_sets(ALL_DATA_FILEPATH, TRAINING_DATA_FILEPATH, VALIDATION_DATA_FILEPATH, VALIDATION_SPLIT)


    # data prep
    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_FILEPATH,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
    VALIDATION_DATA_FILEPATH,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=batch_size,
    )

    # setup model
    base_model = InceptionResNetV2(include_top=False, weights='imagenet') #include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    validation_steps=nb_val_samples/BAT_SIZE,
    class_weight='auto',
    verbose=1)

    # fine-tuning
    # setup_to_finetune(model)
    #
    # history_ft = model.fit_generator(
    #   train_generator,
    #   samples_per_epoch=nb_train_samples,
    #   nb_epoch=nb_epoch,
    #   validation_data=validation_generator,
    #   nb_val_samples=nb_val_samples,
    #   class_weight='auto')

    model.save(args.output_model_file)

    #if args.plot:
    #    plot_training(history_tl)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__=="__main__":
    a = argparse.ArgumentParser()
    # a.add_argument("--train_dir")
    # a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()

    train(args)
