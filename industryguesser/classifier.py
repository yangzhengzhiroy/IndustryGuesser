# -*- coding: UTF-8 -*-
"""
This module creates model object.
"""
import os
import logging
import numpy as np
from subprocess import Popen, PIPE
from .utils import log_config, setup_logging
from industryguesser import PARENT_DIR, ind_cutoff
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.layers import Embedding, Conv1D, Dense, MaxPooling1D, Dropout, Flatten
from .encoder import KerasBatchGenerator, CompanyEncoder, IndustryEncoder


setup_logging(log_config)
logger = logging.getLogger(__name__)


class SimpleCNN(object):
    """ Simple CNN model. """
    _classifier_weights_file_name = 'model_weights.h5'
    _classifier_graph_file_name = 'model_graph.json'
    _classifier_weights_next_name = 'model_weights_next.h5'
    _classifier_graph_next_name = 'graph_next.json'
    _classifier_weights_path = os.path.join(PARENT_DIR, 'models', _classifier_weights_file_name)
    _classifier_graph_path = os.path.join(PARENT_DIR, 'models', _classifier_graph_file_name)
    _classifier_weights_next_path = os.path.join(PARENT_DIR, 'models', _classifier_weights_next_name)
    _classifier_graph_next_path = os.path.join(PARENT_DIR, 'models', _classifier_graph_next_name)

    def __init__(self, lower=True, pad_size=18, padding='post', embedding_size=256, filters=128,
                 kernel_size=3, pool_size=3, cnn_dropout=0.2, optimizer='adam', loss='binary_crossentropy',
                 metrics=None):
        self._pad_size = pad_size
        self._embedding_size = embedding_size
        self._filters = filters
        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._cnn_dropout = cnn_dropout
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics if metrics else ['accuracy']
        self._com_encoder = CompanyEncoder(lower, pad_size, padding)
        self._vocab_size = None
        self._ind_encoder = IndustryEncoder()
        self._model = None

    def _encode_company(self, companies, fit=False):
        """ Encode the input names with NameEncoder. """
        if fit:
            self._com_encoder.fit(companies)
        encoded_companies = self._com_encoder.encode(companies)
        self._vocab_size = self._com_encoder.vocab_size

        return encoded_companies

    def _encode_industry(self, industries, fit=False):
        """ Encode the input genders with GenderEncoder. """
        if fit:
            self._ind_encoder.fit(industries)

        encoded_industries = self._ind_encoder.encode(industries)

        return encoded_industries

    def train(self, companies, industries, split_rate=0.2, batch_size=128, patience=5,
              model_weight_path=_classifier_weights_path, model_graph_path=_classifier_graph_path,
              save_best_only=True, save_weights_only=True, epochs=100):
        """ Train the LSTM model. """
        companies = self._encode_company(companies, True)
        industries = self._encode_industry(industries, True)
        X_train, X_valid, y_train, y_valid = train_test_split(companies, industries, test_size=split_rate)
        valid_batch_size = min(batch_size, len(X_valid) // 3)
        train_gtr = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_gtr = KerasBatchGenerator(X_valid, y_valid, valid_batch_size)

        earlystop = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(model_weight_path, save_best_only=save_best_only,
                                     save_weights_only=save_weights_only)
        history = History()

        model = Sequential()
        model.add(Embedding(input_dim=self._vocab_size, output_dim=self._embedding_size, input_length=self._pad_size))
        model.add(Conv1D(self._filters, self._kernel_size, activation='relu'))
        model.add(MaxPooling1D(self._pool_size))
        model.add(Dropout(rate=self._cnn_dropout))
        model.add(Flatten())
        model.add(Dense(self._ind_encoder.class_size, activation='sigmoid'))
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

        model.fit_generator(train_gtr.generate(), len(X_train) // batch_size, epochs=epochs,
                            validation_data=valid_gtr.generate(), validation_steps=len(X_valid) // valid_batch_size,
                            callbacks=[earlystop, checkpoint, history])
        for epoch in np.arange(0, len(model.history.history['loss'])):
            logger.info(f"Epoch={epoch + 1}, "
                        f"{', '.join(f'{key}={value[epoch]}' for key, value in model.history.history.items())}")

        # Save the model structure.
        with open(model_graph_path, 'w') as f:
            f.write(model.to_json())

        # Load the trained model.
        self._model = model

    def load(self, model_weights_path=_classifier_weights_path, model_graph_path=_classifier_graph_path):
        """ Load the existing master model. """
        K.clear_session()
        with open(model_graph_path, 'r') as f:
            model_graph = f.read()
        self._model = model_from_json(model_graph)
        self._model.load_weights(model_weights_path)

    def update(self, companies, industries, split_rate=0.2, batch_size=64, patience=1,
               model_weights_next_path=_classifier_weights_next_path, model_graph_next_path=_classifier_graph_next_path,
               save_best_only=True, save_weights_only=True, epochs=2):
        """ This function keep the original model, update the model and save it as default model. """
        companies = self._encode_company(companies)
        industries = self._encode_industry(industries)
        X_train, X_valid, y_train, y_valid = train_test_split(companies, industries, test_size=split_rate)
        valid_batch_size = min(batch_size, len(X_valid) // 3)
        train_gtr = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_gtr = KerasBatchGenerator(X_valid, y_valid, valid_batch_size)

        earlystop = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(model_weights_next_path, save_best_only=save_best_only,
                                     save_weights_only=save_weights_only)
        history = History()

        if not self._model:
            self.load()

        self._model.fit_generator(train_gtr.generate(), len(X_train) // batch_size, epochs=epochs,
                                  validation_data=valid_gtr.generate(), validation_steps=len(X_valid) // valid_batch_size,
                                  callbacks=[earlystop, checkpoint, history])
        for epoch in np.arange(0, len(self._model.history.history['loss'])):
            logger.info(f"Epoch={epoch + 1}, "
                        f"{', '.join(f'{key}={value[epoch]}' for key, value in self._model.history.history.items())}")

        # Save the model structure.
        with open(model_graph_next_path, 'w') as f:
            f.write(self._model.to_json())

    def overwrite(self):
        """This function copy the next model version to overwrite the current version."""
        move_file = Popen(f'cp {self._classifier_weights_next_path} {self._classifier_weights_path}; '
                          f'cp {self._classifier_graph_next_path} {self._classifier_graph_path}',
                          shell=True, stdout=PIPE, executable='/bin/bash')
        move_file.communicate()

    def predict(self, companies, return_prob=False, cutoff=ind_cutoff):
        """ This function predicts the gender with given names. """
        if not self._model:
            self.load()
        companies = self._encode_company(companies)
        y_pred_prob = self._model.predict(companies)
        y_pred_prob_max = np.max(y_pred_prob, axis=1)
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        y_pred_class = self._ind_encoder.decode(y_pred_class)
        y_pred_class = np.where(y_pred_prob_max >= cutoff, y_pred_class, None)
        if return_prob:
            return [{'industry': pred, 'prob': [{key: value} for key, value in zip(self._ind_encoder.classes, prob)]}
                    for pred, prob in zip(y_pred_class, y_pred_prob)]
        else:
            return y_pred_class.tolist()
