import sys

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

batch_size = 128
num_classes = 10
epochs = 20
np.argmax(predictions[0])
# input image dimensions
image_rows, image_cols = 28, 28

# Load and split data into train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, image_rows, image_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, image_rows, image_cols)
    input_shape = (1, image_rows, image_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], image_rows, image_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], image_rows, image_cols, 1)
    input_shape = (image_rows, image_cols, 1)


def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model():
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.adam(),
                metrics=['accuracy'])

def train_model(model, x_train, y_train, x_test, y_test):
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test), callbacks=callbacks_list)

def test_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss = ', score[0])
    print('Test accuracy = ', score[1])

def load_model(model, filepath):
    model.load_weights(filepath)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    model = create_model()
    compile_model()
    if (len(sys.argv) == 1):
        train_model(model, x_train, y_train, x_test, y_test)
    else:
        filepath = sys.argv[1]
        load_model(model, filepath)
    test_model(model, x_test, y_test)
