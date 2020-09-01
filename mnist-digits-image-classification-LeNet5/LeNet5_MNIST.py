import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LeNet5(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=[3, 3], strides=1, padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 3], strides=1, padding='valid')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=84, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
train_label = train_label.astype(np.int32)
test_data = np.expand_dims(test_data.astype(np.float32) / 255.0, axis=-1)
test_label = test_label.astype(np.int32)
model = LeNet5()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.2, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=6)
history = model.fit(train_data, train_label, epochs=20, batch_size=128, verbose=2, validation_split=0.1,
                    callbacks=[reduce_lr, early_stopping])
model.save("LeNet5_MNIST")
model.evaluate(test_data, test_label, verbose=2)
plt.plot(history.epoch, history.history["sparse_categorical_accuracy"], 'r--', label='sparse_categorical_accuracy')
plt.plot(history.epoch, history.history["val_sparse_categorical_accuracy"], label='val_sparse_categorical_accuracy')
plt.legend()
plt.show()
