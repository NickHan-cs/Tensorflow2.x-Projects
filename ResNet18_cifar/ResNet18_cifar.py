import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num, kernel_size=[3, 3], strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_num, kernel_size=[3, 3], strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.layers.Conv2D(filter_num, kernel_size=[1, 1], strides=stride, padding='valid')
        else:
            self.downsample = lambda x: x

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(inputs)
        x = tf.keras.layers.add([x, identity])
        x = tf.nn.relu(x)
        return x


class ResNet(tf.keras.Model):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(64, kernel_size=[3, 3], strides=1, padding='valid')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1, padding='same')
        self.resBlocks1 = self.build_ResBlocks(64, 2)
        self.resBlocks2 = self.build_ResBlocks(128, 2, stride=2)
        self.resBlocks3 = self.build_ResBlocks(256, 2, stride=2)
        self.resBlocks4 = self.build_ResBlocks(512, 2, stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resBlocks1(x)
        x = self.resBlocks2(x)
        x = self.resBlocks3(x)
        x = self.resBlocks4(x)
        x = self.avgpool(x)
        x = self.dense(x)
        return x

    def build_ResBlocks(self, filter_num, block_num, stride=1):
        resBlocks = tf.keras.Sequential()
        resBlocks.add(ResBlock(filter_num, stride))
        for _ in range(1, block_num):
            resBlocks.add(ResBlock(filter_num, stride=1))
        return resBlocks


(train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar100.load_data()
train_data = train_data.astype(np.float32) / 255.0
train_label = train_label.astype(np.int32)
num_example = train_data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
train_data = train_data[arr]
train_label = train_label[arr]
test_data = test_data.astype(np.float32) / 255.0
test_label = test_label.astype(np.int32)
model = ResNet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
tf.keras.backend.set_learning_phase(1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.2, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=6)
history = model.fit(train_data, train_label, batch_size=256, epochs=20, verbose=2, validation_split=0.1,
                    callbacks=[reduce_lr, early_stopping])
model.save("ResNet_cifar100_model")
tf.keras.backend.set_learning_phase(0)
model.evaluate(test_data, test_label, verbose=2)
plt.plot(history.epoch, history.history["sparse_categorical_accuracy"], 'r--', label='sparse_categorical_accuracy')
plt.plot(history.epoch, history.history["val_sparse_categorical_accuracy"], label='val_sparse_categorical_accuracy')
plt.legend()
plt.show()
