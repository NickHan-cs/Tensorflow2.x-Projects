import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import matplotlib.pyplot as plt


def get_words(id_list):
    return ' '.join([id2word.get(i, "<UNK>") for i in id_list])


def SimpleRNN():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=200, input_length=maxlen),
        tf.keras.layers.SimpleRNN(64, return_sequences=False),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def GRU():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=200, input_length=maxlen),
        tf.keras.layers.GRU(64, return_sequences=False),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def LSTM():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=200, input_length=maxlen),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def BiLSTM():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=200, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


imdb = tf.keras.datasets.imdb
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
# min(word_index.values()) = 1
word2id = {k: (v + 3) for k, v in word_index.items()}
word2id['<PAD>'] = 0  # 短句子补长的标识符
word2id['<START>'] = 1
word2id['<UNK>'] = 2  # 词表中未出现过的词
word2id['UNUSED'] = 3
id2word = {v: k for k, v in word2id.items()}
# print(train_x[0])
# print(get_words(train_x[0]))
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_x, value=word2id['<PAD>'],
                                                           padding='post', truncating='post', maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_x, value=word2id['<PAD>'],
                                                          padding='post', truncating='post', maxlen=256)
vocab_size = len(word2id)
maxlen = 256
RNN = BiLSTM()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
history = RNN.fit(train_data, train_y, batch_size=64, epochs=25, validation_split=0.2, verbose=2,
                  callbacks=[reduce_lr, early_stopping])
RNN.save("BiLSTM_imdb_model1")
RNN.evaluate(test_data, test_y, verbose=2)
plt.plot(history.epoch, history.history["accuracy"], 'r--', label='accuracy')
plt.plot(history.epoch, history.history["val_accuracy"], label='val_accuracy')
plt.legend()
plt.show()
