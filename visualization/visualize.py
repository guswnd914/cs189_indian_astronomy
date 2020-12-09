import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import DataArrangeTool, PulsatingEpicycleModel
from sklearn import metrics

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 18})


def visualize_epicycle(planet, start=0, end=-1):
    data = DataArrangeTool(planet)
    data.set_parameters()

    x, y = data.dist_x[start:end], data.dist_y[start:end]
    R = np.sqrt(np.power(x, 2) + np.power(y, 2))
    mean = np.mean(R)

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_xlim(-np.max(R), np.max(R))
    ax.set_ylim(-np.max(R), np.max(R))

    circle = plt.Circle((0., 0), mean, fill=False, linewidth=2.5)

    ax.scatter(x, y)
    ax.add_artist(circle)
    ax.plot([0], [0], marker='o')
    plt.xlabel("x-axis distance (au)")
    plt.ylabel("y-axis distance (au)")

    plt.title(f'epicycle movement of {planet}')
    plt.savefig(f'epicycle movement of {planet} for 5000 days')
    plt.show()


def void():
    astros = ['sun', 'moon', 'mercury', 'mars', 'venus', 'jupiter', 'saturn']
    n = 200
    for p in astros:
        visualize_epicycle(p, 0, 5000)
        pred = PulsatingEpicycleModel(p)
        pred.set_parameters()
        pred.split_test_train(axis='y')
        pred.generate_sampling(n)
        pred.optimize_fourier()
        y_pred = pred.predict_position()

        print(f"covariance : {pred.covariance}")
        print(f"mean absolute error: {metrics.mean_absolute_error(pred.y_t, y_pred)}")
        print(f"mean squared error: {metrics.mean_squared_error(pred.y_t, y_pred)}")
        print(f"explained variance score: {metrics.explained_variance_score(pred.y_t, y_pred)}")
        print(f"median absolute error: {metrics.median_absolute_error(pred.y_t, y_pred)}")
        print(f"r2 score: {metrics.r2_score(pred.y_t, y_pred)}")

        cov_min, cov_max = np.mean(pred.covariance) - np.std(pred.covariance), np.mean(pred.covariance) + np.std(
            pred.covariance)
        heatmap = sns.heatmap(pred.covariance, vmin=cov_min, vmax=cov_max, annot=True, cmap='BrBG', center=0,
                              linewidth=3, linecolor='k')
        heatmap.set_title(f"Covariance Heatmap of Outcome for {p} at N = {n}", fontdict={'fontsize': 18}, pad=12);
        fig = heatmap.get_figure()
        fig.savefig(f"Covariance Heatmap of Outcome for {p} at N = {n}")

        plt.figure(figsize=(20, 10))

        plt.plot(pred.x_, pred.y_, "g-", label="train", linewidth=2)
        plt.plot(pred.x_t, y_pred, "r-", label="prediction", linewidth=2)
        plt.plot(pred.x_t, pred.y_t, "b", label='test')
        plt.legend(loc="best")
        plt.title(f"FFT of {p} in its distance at N = {n}")
        plt.xlabel("time in days")
        plt.ylabel("distance in au")
        plt.savefig(f"FFT of {p} in its distance at N = {n}")
        plt.show()


if __name__ == "__main__":
    void()
