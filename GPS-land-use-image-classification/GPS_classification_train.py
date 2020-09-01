import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import time

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
    net_model = models.Model(inputs, outputs, name='net')
    return net_model


model = net()
# 记录模型训练日志
log_dir = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def img_rotate(img):
    angle = np.random.choice(['up_down', 'up_down_left_right', 'left_right', 'none'])
    if angle == 'up_down':
        img = tf.image.flip_up_down(img)
    elif angle == 'up_down_left_right':
        img = tf.image.flip_up_down(img)
        img = tf.image.flip_left_right(img)
    elif angle == 'left_right':
        img = tf.image.flip_left_right(img)
    elif angle == 'none':
        pass
    return img


def convert(img, label):
    image = tf.image.convert_image_dtype(img, tf.float32)
    image = tf.image.resize(image, size=[32, 32])
    return image, label


def data_augement(image, label):
    image, label = convert(image, label)
    image = img_rotate(image)
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=0.5)
    # 随机设置图片的对比度
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # 随机设置图片的色度
    image = tf.image.random_hue(image, max_delta=0.3)
    # 随机设置图片的饱和度
    image = tf.image.random_saturation(image, lower=0.2, upper=1.8)
    return image, label


raw_train, metadata = tfds.load(
    'uc_merced',  # 数据集名称,gps拍摄的实景
    split=['train'],  # 仅提供train数据
    with_info=True,  # 这个参数和metadata对应
    as_supervised=True,  # 这个参数的作用时返回tuple形式的(input, label),举个例子,raw_test=tuple(input, label)
    shuffle_files=True,
    data_dir=os.path.abspath(os.path.dirname(__file__)) + os.sep + 'tensorflow_datasets'
)
batch_size = 16
shuffle_buffer_size = 2520
train_batches = raw_train[0].shuffle(shuffle_buffer_size).map(data_augement).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_batches, epochs=50, callbacks=[tensorboard_callback])
keras_file = os.path.abspath(os.path.abspath(__file__)) + os.sep + time.strftime("%Y-%m-%d-%H:%M:%S",
                                                                                 time.localtime()) + '-test.h5'
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
