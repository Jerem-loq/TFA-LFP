"""
Vrest distribution plot
Data from Allan Brain Institute
Interactive Normalized Double Gaussian Distribution

Made by Jérémie Loquet 01/2025
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest
from matplotlib.widgets import Slider


def read_data(path):
    """
    Read .csv file.
    This one works only with the data from Allan Brain Institute.
    Change column name to get the distribution you want.
    :param path: path of the data file
    :return: data to plot.
    """
    df = pd.read_csv(path)
    df_filtered = df[["ef__vrest"]]
    df_filtered = df_filtered.dropna()

    return df_filtered


def Normality(dataset):
    """
    Perform Kolmogorov-Smirnov Test.
    Can be adapted to perform Shapiro-Wilk instead.
    :param dataset: data to test
    :return: Statistical value of the test, p-val.
    """
    result = kstest(dataset, 'norm')
    stat_ks = result.statistic[0]
    p_ks = result.pvalue[0]
    print(f"\nKolmogorov-Smirnov Test: \tStatistics = {stat_ks}, \tp-value = {p_ks}.")

    return stat_ks, p_ks


def Visualization(dataset):
    """
    Plot histogram and Kernel-Density Estimation
    :param dataset: Data to plot
    """
    values = dataset["ef__vrest"].values
    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    sns.histplot(values, kde=True, ax=ax[1], legend=False)
    ax[1].set_title(r"$V_{rest}$ distribution with KDE")
    stats.probplot(values, dist='norm', plot=ax[0])
    ax[0].set_title("Q-Q Plot")
    plt.tight_layout()
    plt.show()


def P(X, mu1, mu2, sigma1, sigma2, A):
    """
    Bimodal Normal distribution equation.
    """
    equation = (
            A / (np.sqrt(2 * np.pi * sigma1)) * np.exp(- (X - mu1) ** 2 / sigma1 ** 2) +
            (1 - A) / (np.sqrt(2 * np.pi * sigma2)) * np.exp(- (X - mu2) ** 2 / sigma2 ** 2)
    )

    return equation


def Double_Gaussian(x, y):
    """
    Plot Interactive Bimodal Normal curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.4)
    line, = ax.plot(x, y)
    plt.xlabel(r"$V_{rest}$")
    plt.ylabel("Probability Density")
    plt.title("Normalized Bimodal Gaussian Distribution")

    axcolor = 'lightgoldenrodyellow'
    ax_mu1 = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_mu2 = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_sigma1 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_sigma2 = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_A = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    slider_mu1 = Slider(ax_mu1, r'$\mu_1$', -85, -55, valinit=mu1_init)
    slider_mu2 = Slider(ax_mu2, r'$\mu_2$', -85, -55, valinit=mu2_init)
    slider_sigma1 = Slider(ax_sigma1, r'$\sigma_1$', 1, 10, valinit=sigma1_init)
    slider_sigma2 = Slider(ax_sigma2, r'$\sigma_2$', 1, 10, valinit=sigma2_init)
    slider_A = Slider(ax_A, 'A', 0.01, 0.99, valinit=A_init)

    def update(val):
        mu1 = slider_mu1.val
        mu2 = slider_mu2.val
        sigma1 = slider_sigma1.val
        sigma2 = slider_sigma2.val
        A = slider_A.val
        y_new = P(x, mu1, mu2, sigma1, sigma2, A)
        line.set_ydata(y_new)
        ax.set_ylim(0, max(y_new) * 1.1)
        fig.canvas.draw_idle()

    slider_mu1.on_changed(update)
    slider_mu2.on_changed(update)
    slider_sigma1.on_changed(update)
    slider_sigma2.on_changed(update)
    slider_A.on_changed(update)

    plt.show()


if __name__ == "__main__":
    file_path = r''  # Path to .csv file
    data = read_data(file_path)

    # Compute the Bimodal Normal Distribution
    E_l = np.linspace(-85, -55, 100)
    mu1_init, mu2_init = -72, -61
    sigma1_init, sigma2_init = 6, 3
    A_init = 0.5

    # Test normality and Plot the data
    Normality(data)
    Visualization(data)

    # Plot the interactive Bimodal Normal Distribution
    P_vector = P(E_l, mu1_init, mu2_init, sigma1_init, sigma2_init, A_init)
    Double_Gaussian(E_l, P_vector)


