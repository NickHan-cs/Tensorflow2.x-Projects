import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt

layers = tf.keras.layers
models = tf.keras.models


def net(input_shape=(32, 32, 3), classes=21):
    input_img = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=64,
                      kernel_size=(5, 5),
                      activation='relu',
                      padding='valid',
                      name='conv1')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(filters=256,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='valid',
                      name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=128,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='valid',
                      name='conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=classes, activation='softmax', name='predicitions')(x)
    inputs = input_img
    outputs = x
    model = models.Model(inputs, outputs, name='net')
    return model


model = net()


def convert(img, label):
    image = tf.image.convert_image_dtype(img, tf.float32)
    image = tf.image.resize(image, size=[32, 32])
    return image, label


raw_test, metadata = tfds.load(
    'uc_merced',  # 数据集名称,gps拍摄的实景
    split=['train'],  # 仅提供train数据
    with_info=True,  # 这个参数和metadata对应
    as_supervised=True,  # 这个参数的作用时返回tuple形式的(input, label),举个例子,raw_test=tuple(input, label)
    shuffle_files=True,
    data_dir=os.path.abspath(os.path.dirname(__file__)) + os.sep + 'tensorflow_datasets'
)

model.load_weights(os.path.abspath(os.path.dirname(__file__)) + os.sep + "2020-09-01-07:18:47-test.h5")

get_label_name = metadata.features['label'].int2str

for image, label in raw_test[0].take(5):
    image, label = convert(image, label)
    predict = np.argmax(model.predict(np.expand_dims(image, axis=0)))
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(predict))
    plt.show()
