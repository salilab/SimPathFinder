import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from gensim.models.fasttext import FastText as FT_gensim
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from gensim.models import FastText
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import kerastuner
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from models import ModelVectors


class DNNModel(ModelVectors):
    def __init__(self):
        super().__init__(data_dir='labeldata/',
                         model_dir='models/',
                         categories=['Detoxification',
                                     'Activation',
                                     'Biosynthesis',
                                     'Degradation',
                                     'Energy',
                                     'Glycan',
                                     'Macromolecule'])
        self.fetchModel()
        self.X_train = self.trainLSTM.EC.to_list()
        self.X_test = self.testLSTM.EC.to_list()
        self.X_valid = self.validateLSTM.EC.to_list()
        self.Y_train = self.trainLSTM[self.categories].values
        self.Y_valid = self.validateLSTM[self.categories].values
        self.Y_test = self.testLSTM[self.categories].values

    def cleanTextChar(self, num_words=10000,
                      oov_token='<OOV>',
                      padding='post',
                      train=[],
                      test=[],
                      valid=[]):
        if not train:
            train = [' '.join(i) for i in self.X_train]
        if not valid:
            valid = [' '.join(i) for i in self.X_valid]
        if not test:
            test = [' '.join(i) for i in self.X_test]
        self.token = Tokenizer(num_words=num_words,
                               oov_token=oov_token, char_level=True)
        self.token.fit_on_texts(train+valid+test)
        self.word_index = self.token.word_index
        seq_train = self.token.texts_to_sequences(train)
        self.paddedTrain = pad_sequences(seq_train, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valid = self.token.texts_to_sequences(valid)
        self.paddedValid = pad_sequences(seq_valid, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_test = self.token.texts_to_sequences(test)
        self.paddedTest = pad_sequences(seq_test, maxlen=len(
            max(train, test, valid)), padding=padding)

        return self.paddedTrain, self.paddedTest, self.paddedValid, self.token, self.word_index

    def createEmbMatrix(self,):
        self.embedding_matrix_ft = np.random.random(
            (len(self.token.word_index) + 1, self.model.vector_size))
        pas = 0
        for word, i in self.token.word_index.items():
            try:
                self.embedding_matrix_ft[i] = self.model.wv[word]
            except IndexError:
                pas += 1

    def getClassWeights(self,):
        number_dim = np.shape(self.Y_train)[1]
        self.weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            self.weights[i] = compute_class_weight(
                'balanced', [0., 1.], self.Y_train[:, i])
        self.class_dict = {i: 0.1*self.weights[i, 1]
                           for i in range(len(self.categories))}
        return self.weights, self.class_dict

    def generateSampleWeights(self, ):
        self.sample_weights = np.array([1 if sum(self.Y_train[i, :]) == 1 else sum(self.Y_train[i, :])
                                        for i in range(self.Y_train.shape[0])])
        return self.sample_weights

    def defineModel(self,):
        self.LSTMmodel = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.word_index)+1,
                                      output_dim=300,
                                      weights=[self.embedding_matrix_ft],
                                      trainable=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(160)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(182, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(480, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.Y_train.shape[1], activation='sigmoid')
        ])
        return self.LSTMmodel

    def defineModelFinal(self,):
        self.LSTMmodel = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.word_index)+1,
                                      output_dim=300,
                                      weights=[self.embedding_matrix_ft],
                                      trainable=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu',),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.Y_train.shape[1], activation='sigmoid')
        ])

        return self.LSTMmodel

    def defineModelFinalR(self,):
        self.LSTMmodel = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.word_index)+1,
                                      output_dim=300,
                                      weights=[self.embedding_matrix_ft],
                                      trainable=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='relu',),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.Y_train.shape[1], activation='sigmoid')
        ])
        return self.LSTMmodel

    def fitModel(self, num_epochs=3):
        self.buildModel()
        self.getClassWeights()
        self.LSTMmodel.compile(loss='binary_crossentropy',
                               optimizer=tf.keras.optimizers.Adam(
                                   learning_rate=1e-3),
                               metrics=['accuracy'], class_weight=self.class_dict)
        self.history = self.LSTMmodel.fit(self.paddedTrain, self.Y_train,
                                          epochs=num_epochs,
                                          validation_data=(self.paddedTest, self.Y_test))

    def fitModelEarlyStopping(self, num_epochs=500):
        self.LSTMmodel = self.defineModelFinalR()
        self.LSTMmodel.summary()
        self.getClassWeights()
        es = EarlyStopping(monitor='loss', mode='min',  patience=3)

        self.LSTMmodel.compile(loss='binary_crossentropy', optimizer='adam',
                               metrics=[tf.keras.metrics.BinaryAccuracy(
                                   name='accuracy')],
                               class_weight=self.class_dict)
        self.history = self.LSTMmodel.fit(self.paddedTrain, self.Y_train,
                                          epochs=num_epochs,
                                          validation_data=(
                                              self.paddedTest, self.Y_test),
                                          callbacks=[es], batch_size=500)
        self.LSTMmodel.save('dnnmodels/my_model_LSTMFhalf.h5')

    def plotAccLoss(self,):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def parameterizeModel(self, hp,):
        model = tf.keras.Sequential()
        self.getClassWeights()
        self.generateSampleWeights()
        model.add(layers.Embedding(input_dim=len(self.word_index)+1,
                                   output_dim=300,
                                   weights=[self.embedding_matrix_ft], trainable=True))

        model.add(layers.Dropout(0.5)),
        model.add(layers.Bidirectional(layers.LSTM(units=hp.Int('unitsLSTM',
                                                                min_value=8,
                                                                max_value=512,
                                                                step=32))))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5)),
        model.add(layers.Dense(units=hp.Int('unitsDense',
                                            min_value=16,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(self.Y_train.shape[1], activation='sigmoid'))
        model.compile(batch_size=500,
                      optimizer=tf.keras.optimizers.Adam(
                          hp.Choice('learning_rate',
                                    values=[1e-3])),
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(
                          name='accuracy')],
                      class_weight=self.class_dict,
                      sample_weight=self.sample_weights)
        return model

    def runParameterizeModel(self,):
        self.tuner = RandomSearch(
            self.parameterizeModel,
            objective=kerastuner.Objective("val_loss", direction="min"),
            max_trials=5,
            executions_per_trial=3,
            directory='my_dir',
            project_name='doubleSample')
        self.tuner.search_space_summary()
        self.tuner.search(self.paddedTrain, self.Y_train,
                          epochs=5, batch_size=500,
                          validation_data=(self.paddedTest, self.Y_test))
        self.tuner.results_summary()
