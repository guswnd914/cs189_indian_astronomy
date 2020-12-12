from predict_epicycle import DataArrangeTool, PulsatingEpicycleModel

astros = ['sun', 'moon', 'mercury', 'mars', 'venus', 'jupiter', 'saturn']


def predict(astro:list = None, n=300):
    """Predict and evaluate the model, and save the result in data directory.

    :param astro: list of planets to predict
    :param n: number of data points
    :return:
    """
    if astro is None:
        astro = astros

    pred_data = []
    eval_data = []
    for a in astro:
        pred_curr = {'time': [],
                      a: {'[true_dist, pred_dist]': {}}}
        eval_curr = {a: {}}
        pred = PulsatingEpicycleModel(a)
        pred.set_parameters()
        for axis in 'xy':
            pred.split_test_train(axis=axis)
            pred.generate_sampling(n)
            pred.optimize_fourier()
            pred.predict_position()

            pred_curr[a]['[true_dist, pred_dist]'][f'{axis}_dist'] = pred.fetch_prediction()
            pred_curr[a][f'{axis}_axis'] = pred.fetch_prediction(predict=True)
            eval_curr[a][f'{axis}_axis'] = pred.evaluate()
        pred_curr[a]['time'] = pred.fetch_prediction(predict=False)

        pred_data.append(pred_curr)
        eval_data.append(eval_curr)

    DataArrangeTool.save_data(pred_data, evaluation=False)
    DataArrangeTool.save_data(eval_data, evaluation=True)

