from tensorflow.keras.datasets import cifar100
from tensorflow import keras

from dl_project.project import Project

import tensorflow as tf

input_shape = (32, 32)
label_num = 50000
epochs = 50

project = Project("resnet_cifar100")

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


with tf.device('/gpu:1'):
    model = project.build('resnet', input_shape, label_num)

    sgd = keras.optimizers.SGD(momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    project.plot_model(model)
    project.load_ckpt(model)

    model.fit(
        x=x_train, y=y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=project.callbacks
    )
