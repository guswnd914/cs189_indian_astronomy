from process_data import generate
from predict_epicycle import DataArrangeTool, PulsatingEpicycleModel

import numpy as np
import os
import json
from pathlib import Path


def save_data(data: np.array, evaluation: bool = False) -> None:
    """Save predicted output in data directory.

    :param evaluation: Determine predicted data or report.
    :param data: Predicted data.
    """
    if evaluation:
        data_path = os.path.join(Path(os.getcwd()).parent, 'evaluation')
        file = 'predicted_data_evaluation.json'
    else:
        data_path = os.path.join(Path(os.getcwd()).parent, 'data')
        file = 'predicted_data.json'
    try:
        with open(os.path.join(data_path, file), 'w') as fw:
            json.dump(data, fw, indent=3)
    except IOError:
        raise Exception("I/O error")


def void():
    generate()

    astros = ['sun', 'moon', 'mercury', 'mars', 'venus', 'jupiter', 'saturn']

    pred_data = []
    eval_data = []
    for a in astros:
        pred = PulsatingEpicycleModel(a)
        pred.set_parameters()
        pred.split_test_train(axis='y')
        pred.generate_sampling(300)
        pred.optimize_fourier()
        y_pred = pred.predict_position()
        pred_data.append(y_pred)
        eval_data.append(pred.evaluate_regression(y_pred, pred.y_t, a))

    save_data(pred_data, evaluation=False)
    save_data(eval_data, evaluation=True)


if __name__ == '__main__':
    void()
