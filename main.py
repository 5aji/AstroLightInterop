

import astrolightinterop.RAPID.rapidmodel as rapidmodel
import logging
import matplotlib.pyplot as plt
import pandas as pd
logging.basicConfig(level=logging.INFO)

training_data = pd.read_csv('data/training_set.csv', index_col=['object_id', 'mjd'])
training_meta = pd.read_csv('data/training_set_metadata.csv', index_col='object_id')
model = rapidmodel.RAPIDModel(training_data, training_meta)
targets, predictions = model.test()
