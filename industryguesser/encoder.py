# -*- coding: UTF-8 -*-
"""
This module prepares the input names and gender label.
"""
import os
import re
import pickle
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from industryguesser import PARENT_DIR


logger = logging.getLogger(__name__)


class KerasBatchGenerator(object):

    def __init__(self, data, label, batch_size):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.current_idx = 0
        self.num_of_batches = len(data) // batch_size

    def generate(self):
        while True:
            if self.current_idx >= self.num_of_batches:
                self.current_idx = 0
            for index in range(self.num_of_batches):
                x, y = self.data[(self.current_idx * self.batch_size):((self.current_idx + 1) * self.batch_size)], \
                       self.label[(self.current_idx * self.batch_size):((self.current_idx + 1) * self.batch_size)]
                self.current_idx += 1
                yield x, y


class IndustryEncoder(object):
    """ Encode the gender to categories. """
    _ind_encoder_file_name = 'industry_encoder.pkl'
    _encoder_path = os.path.join(PARENT_DIR, 'models', _ind_encoder_file_name)

    def __init__(self):
        self._ind_encoder = None
        self._fit = False
        self._load = False

    def fit(self, industries):
        """ Fit the gender label encoder if needed. """
        self._ind_encoder = LabelEncoder()
        self._ind_encoder.fit(list(set(industries)))

        with open(self._encoder_path, 'wb') as f:
            pickle.dump(self._ind_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._fit = True

    def load(self):
        """ Load the pre-fit gender label encoder. """
        with open(self._encoder_path, 'rb') as f:
            self._ind_encoder = pickle.load(f)

        self._load = True

    def encode(self, industries):
        """ Convert gender values to encoded integers. """
        if not self._ind_encoder:
            self.load()

        encoded_industries = self._ind_encoder.transform(industries)
        encoded_industries = to_categorical(encoded_industries)

        return encoded_industries

    def decode(self, y_pred):
        """ Convert gender values to encoded integers. """
        if not self._ind_encoder:
            self.load()

        industries = self._ind_encoder.inverse_transform(y_pred)
        return industries

    @property
    def class_size(self):
        if not self._ind_encoder:
            self.load()

        return len(self._ind_encoder.classes_)

    @property
    def classes(self):
        if not self._ind_encoder:
            self.load()

        return self._ind_encoder.classes_


class CompanyEncoder(object):
    """ Encode the name list into encoded char-to-int 2-D numpy array. """
    _com_encoder_file_name = 'company_encoder.pkl'
    _encoder_path = os.path.join(PARENT_DIR, 'models', _com_encoder_file_name)

    def __init__(self, lower=True, pad_size=21, padding='post'):
        self._lower = lower
        self._com_encoder = None
        self._fit = False
        self._load = False
        self._pad_size = pad_size
        self._padding = padding
        self._stopwords = \
            {'of', 'inc.', 'ltd', 'group', 'the', 'and', 'inc', 'ltd.', 'llc', 'company', 'limited', 'corporation',
             'pvt', 'pvt.', 'for', 'co.', 'at', 'formerly', 'us', 'usa', 'india', 'canada', 'llp', 'a', 'corp', 'co',
             'in', 'u.s.', 'city', 'world', 'china', 'corporate', 'p', 't', 'by', 's.a.', 'sa', 'uk', 'plc', 'm', 'i',
             'e', 'asia', 'europe', 'romania', 'washington', 'enterprise', 'enterprises', 'an', 'la', 'deloitte', 'al',
             'on', 'boston', 'â', 'london', 'subsidiary', 'regional', 'as', 'corps', 'r', 'western', 'africa',
             'singapore', 'pakistan', 'd', 'mexico', 'to', 'groupe', 'georgia', 'o', 'firm', 'mumbai', 'ohio',
             'bangalore', 'delhi', 'illinois', 'llc.', 'ã', 'dubai', 'indian', 'holding', 'j', 'c', 'british',
             'colorado', 'virginia', 'paris', 'pvt.ltd.', 'pte', 'pune', 'indiana', 'l', 'japan', 'canadian',
             'shanghai', 'b', 'european', 'minnesota', 'ca', 'deutsche', 'atlanta', 'houston', 'toronto', 'k',
             'miami', 's.', 'saudi', 'a.', 'pllc', 'g', 'f', 'h', 'n', 'q', 's', 'u', 'v', 'w', 'x', 'y', 'z',
             'missouri', 'jersey', 'england', 'chennai', 'tennessee', 'philadelphia', 'pennsylvania', 'indonesia',
             'asian', 'ireland', 'iowa', '.', 'alabama', 'cambridge', 'israel', 'netherlands', 'detroit', 'seattle',
             'philippines', 'connecticut', 'argentina', '1', 'malaysia', 'venezuela', 'scotland', '-'}

    def text_clean(self, company):
        """ Clean the input name string. """
        try:
            if self._lower:
                company = company.lower()

            company = re.sub('[^\\w \\-"\'.]+', ' ', company)
            company = company.strip().split()
            company = [token for token in company if token not in self._stopwords]
            company = ' '.join(company)
            return company
        except (TypeError, AttributeError) as e:
            logger.exception(f'text_clean [{company}]: {e}')

    def fit(self, companies, num_words=200000):
        """ Fit the new encoder if not loaded. """
        clean_companies = [self.text_clean(company) for company in companies]
        self._com_encoder = Tokenizer(num_words=num_words)
        self._com_encoder.fit_on_texts(clean_companies)

        with open(self._encoder_path, 'wb') as f:
            pickle.dump(self._com_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._fit = True

    def load(self):
        """ Load the fitted encoder. """
        with open(self._encoder_path, 'rb') as f:
            self._com_encoder = pickle.load(f)
        self._load = True

    def encode(self, companies):
        """ Encode all input names. """
        if not self._com_encoder:
            self.load()

        companies = [self.text_clean(company) for company in companies]
        encoded_companies = self._com_encoder.texts_to_sequences(companies)
        encoded_companies = pad_sequences(encoded_companies, maxlen=self._pad_size, padding=self._padding)

        return encoded_companies

    @property
    def vocab_size(self):
        return len(self._com_encoder.word_index)
