# -*- coding: UTF-8 -*-
"""
IndustryGuesser package
"""
import os


__author__ = 'yang zhengzhi'

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ind_cutoff = 0.5

from .api import guess_industry, guess_industries
