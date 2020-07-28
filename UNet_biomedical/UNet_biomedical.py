from google.colab import drive
import os
import numpy as np
import tensorflow as tf
import skimage.transform as trans
import skimage.io as io

def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1, )) if not flag_multi_class else img
        img = np.reshape(img, (1, )+img.shape)
        yield img


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = np.asarray(item[:, :, 0] * 255.0, np.int32)
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


class UNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn6 = tf.keras.layers.BatchNormalization()

        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.conv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(0.5)

        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv9 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3, 3], strides=1, padding='same',
                                            kernel_initializer='he_normal')
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.conv10 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn10 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(0.5)

        self.up1 = tf.keras.layers.UpSampling2D(size=[2, 2])
        self.conv11 = tf.keras.layers.Conv2D(filters=512, kernel_size=[2, 2], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn11 = tf.keras.layers.BatchNormalization()
        self.conv12 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn12 = tf.keras.layers.BatchNormalization()
        self.conv13 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn13 = tf.keras.layers.BatchNormalization()

        self.up2 = tf.keras.layers.UpSampling2D(size=[2, 2])
        self.conv14 = tf.keras.layers.Conv2D(filters=256, kernel_size=[2, 2], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn14 = tf.keras.layers.BatchNormalization()
        self.conv15 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn15 = tf.keras.layers.BatchNormalization()
        self.conv16 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn16 = tf.keras.layers.BatchNormalization()

        self.up3 = tf.keras.layers.UpSampling2D(size=[2, 2])
        self.conv17 = tf.keras.layers.Conv2D(filters=128, kernel_size=[2, 2], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn17 = tf.keras.layers.BatchNormalization()
        self.conv18 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn18 = tf.keras.layers.BatchNormalization()
        self.conv19 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn19 = tf.keras.layers.BatchNormalization()

        self.up4 = tf.keras.layers.UpSampling2D(size=[2, 2])
        self.conv20 = tf.keras.layers.Conv2D(filters=64, kernel_size=[2, 2], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn20 = tf.keras.layers.BatchNormalization()
        self.conv21 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn21 = tf.keras.layers.BatchNormalization()
        self.conv22 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.bn22 = tf.keras.layers.BatchNormalization()
        self.conv23 = tf.keras.layers.Conv2D(filters=2, kernel_size=[3, 3], strides=1, padding='same',
                                             kernel_initializer='he_normal')
        self.conv24 = tf.keras.layers.Conv2D(filters=1, kernel_size=[1, 1], strides=1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        c1 = tf.nn.relu(x)

        x = self.pool1(c1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        c2 = tf.nn.relu(x)

        x = self.pool2(c2)
        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        c3 = tf.nn.relu(x)

        x = self.pool3(c3)
        x = self.conv7(x)
        x = self.bn7(x)
        x = tf.nn.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = tf.nn.relu(x)
        c4 = self.drop1(x)

        x = self.pool4(c4)
        x = self.conv9(x)
        x = self.bn9(x)
        x = tf.nn.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = tf.nn.relu(x)
        x = self.drop2(x)

        x = self.up1(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.concatenate([c4, x], axis=3)
        x = self.conv12(x)
        x = self.bn12(x)
        x = tf.nn.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = tf.nn.relu(x)

        x = self.up2(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.concatenate([c3, x], axis=3)
        x = self.conv15(x)
        x = self.bn15(x)
        x = tf.nn.relu(x)
        x = self.conv16(x)
        x = self.bn16(x)
        x = tf.nn.relu(x)

        x = self.up3(x)
        x = self.conv17(x)
        x = self.bn17(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.concatenate([c2, x], axis=3)
        x = self.conv18(x)
        x = self.bn18(x)
        x = tf.nn.relu(x)
        x = self.conv19(x)
        x = self.bn19(x)
        x = tf.nn.relu(x)

        x = self.up4(x)
        x = self.conv20(x)
        x = self.bn20(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.concatenate([c1, x], axis=3)
        x = self.conv21(x)
        x = self.bn21(x)
        x = tf.nn.relu(x)
        x = self.conv22(x)
        x = self.bn22(x)
        x = tf.nn.relu(x)
        x = self.conv23(x)
        x = tf.nn.relu(x)
        x = self.conv24(x)
        return x


drive.mount('/content/drive')
path = "/content/drive/My Drive"
os.chdir(path)
train_imgs = np.load("/content/drive/My Drive/DeepLearningDatasets/biomedical_image_segmentation/train/image_arr.npy")
train_masks = np.load("/content/drive/My Drive/DeepLearningDatasets/biomedical_image_segmentation/train/mask_arr.npy")
model = UNet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20)
history = model.fit(train_imgs, train_masks, epochs=200, verbose=2, batch_size=4, callbacks=[reduce_lr, early_stopping])
model.save("/content/drive/My Drive/UNet_biomedical")
testGene = testGenerator("/content/drive/My Drive/DeepLearningDatasets/biomedical_image_segmentation/test/img/")
results = model.predict_generator(testGene, verbose=2)
saveResult("/content/drive/My Drive/DeepLearningDatasets/biomedical_image_segmentation/predict/", results)
