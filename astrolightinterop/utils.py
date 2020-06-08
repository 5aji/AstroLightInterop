import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np


def print_test_results(target_list: list[int], predictions_list, use_probabilities=False):
    """
    Prints standard performance metrics for the results
    :param target_list: The list of "true" classes
    :param predictions_list: The list of predicted classes.
    """
    if use_probabilities:
        # The number of classes in the target list should be equal to the length of the probability
        # vector in pred_list
        assert len(np.unique(target_list)) == len(predictions_list[0][0])


class BaseModel:
    pass  # TODO: use basemodel to define consistent API for models


