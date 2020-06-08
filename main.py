

import astrolightinterop.RAPID.rapidmodel as rapidmodel
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
model = rapidmodel.Model('data/training_set.csv', 'data/training_set_metadata.csv')
targets, predictions = model.test()
