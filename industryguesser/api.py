import logging
from .utils import log_config, setup_logging
from .classifier import SimpleCNN
from industryguesser import ind_cutoff


setup_logging(log_config)
logger = logging.getLogger(__name__)
ind_model = SimpleCNN()


def guess_industries(companies, return_prob=False, cutoff=ind_cutoff):
    """ The function guesses industries and probabilities based on company names. """
    try:
        if companies:
            return ind_model.predict(companies, return_prob, cutoff)
    except Exception as e:
        logger.exception(f'predict_industries: {e}')


def guess_industry(company, return_prob=False, cutoff=ind_cutoff):
    try:
        output = guess_industries([company], return_prob, cutoff)
        if output:
            return output[0]
        else:
            return None
    except Exception as e:
        logger.exception(f'predict_gender: {e}')
