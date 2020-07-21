import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_img(path):
    imgs = []
    labels = []
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    for index, i in enumerate(cate):
        for j in os.listdir(i):
            im = cv2.imread(i + '/' + j)
            img1 = cv2.resize(im, (171, 171)) / 255
            imgs.append(img1)
            labels.append(index)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def data_augmentation(train_imgs, train_labels):
    final_imgs = []
    final_labels = []
    for i in range(train_imgs.shape[0]):
        img = train_imgs[i]
        label = train_labels[i]
        final_imgs.append(img)
        final_labels.append(label)
        img1 = tf.image.flip_left_right(img).numpy()
        final_imgs.append(img1)
        final_labels.append(label)
        img2 = tf.image.flip_up_down(img).numpy()
        final_imgs.append(img2)
        final_labels.append(label)
    return np.asarray(final_imgs, np.float32), np.asarray(final_labels, np.int32)


class AlexNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=[9, 9],
            strides=3,
            activation=tf.nn.relu,
            padding='valid'
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, padding='valid')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[5, 5],
            strides=1,
            activation=tf.nn.relu,
            padding='same'
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, padding='valid')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            filters=192,
            kernel_size=[3, 3],
            strides=1,
            activation=tf.nn.relu,
            padding='same'
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=192,
            kernel_size=[3, 3],
            strides=1,
            activation=tf.nn.relu,
            padding='same'
        )
        self.conv5 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            strides=1,
            activation=tf.nn.relu,
            padding='same'
        )
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2, padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=5, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.dense3(x)
        return x


path = r'D:/TensorFlow_datasets/flower_photos/'
data, label = read_img(path)
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
ratio = 0.8
s = np.int(num_example * ratio)
train_data = data[:s]
train_label = label[:s]
test_data = data[s:]
test_label = label[s:]
train_data, train_label = data_augmentation(train_data, train_label)
num_example = train_data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
train_data = train_data[arr]
train_label = train_label[arr]
model = AlexNet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.2, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=6)
history = model.fit(train_data, train_label, batch_size=32, epochs=25, verbose=2, validation_split=0.1,
                    callbacks=[reduce_lr, early_stopping])
model.save('AlexNet_flowers_model1')
model.evaluate(test_data, test_label, verbose=2)
plt.plot(history.epoch, history.history["sparse_categorical_accuracy"], 'r--', label='sparse_categorical_accuracy')
plt.plot(history.epoch, history.history["val_sparse_categorical_accuracy"], label='val_sparse_categorical_accuracy')
plt.legend()
plt.show()
