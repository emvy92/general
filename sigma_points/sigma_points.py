import numpy.linalg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import math
from matplotlib.animation import FuncAnimation


def update(frame):
    global sigma_a, sigma_b
    corr = frame/100.0
    cov = corr*sigma_a*sigma_b
    mean = np.matrix([[0.0], [0.0]])
    A = np.matrix([[sigma_a*sigma_a, cov], [cov, sigma_b*sigma_b]])

    # sigma points via eigen decomposition and confidence ellipse
    sp_eig, center, width, height, angle = calculate_sp_eig(mean, A)

    # sigma points via cholesky decomposition
    sp_chol = calculate_sp_chol(mean, A)

    # update plots
    global ell, plot_chol, plot_eig
    ell.center = center
    ell.angle = angle
    ell.width = width
    ell.height = height
    plot_chol.set_data(sp_chol[0, :].tolist()[0], sp_chol[1, :].tolist()[0])
    plot_eig.set_data(sp_eig[0, :].tolist()[0], sp_eig[1, :].tolist()[0])

    return ell, plot_chol, plot_eig


def calculate_sp_eig(mean, cov):
    val_eig, vec_eig = numpy.linalg.eig(cov)
    sp_eig = mean
    for i in range(0, len(val_eig)):
        sp_delta = vec_eig[:, i]*np.sqrt(val_eig[i])
        sp_eig = np.hstack((sp_eig, mean+sp_delta))
        sp_eig = np.hstack((sp_eig, mean-sp_delta))
    cov_eig = np.matrix([[0.0], [0.0]])
    for i in range(0, sp_eig.shape[1]):
        cov_eig = cov_eig + sp_eig[:, i]*sp_eig[:, i].transpose()
    cov_eig = 0.5*cov_eig

    # confidence ellipse
    center = [mean[0], mean[1]]
    width = 2*np.sqrt(val_eig[0])
    height = 2*np.sqrt(val_eig[1])
    angle = np.arctan2(vec_eig[1, 0], vec_eig[0, 0]) * (180/math.pi)

    return sp_eig, center, width, height, angle


def calculate_sp_chol(mean, cov):
    chol = numpy.linalg.cholesky(cov)
    mean = np.matrix([[0.0], [0.0]])
    sp_chol = mean
    for i in range(0, chol.shape[1]):
        sp_chol = np.hstack((sp_chol, mean+chol[:, i]))
        sp_chol = np.hstack((sp_chol, mean-chol[:, i]))
    cov_chol = np.matrix([[0.0], [0.0]])
    for i in range(0, sp_chol.shape[1]):
        cov_chol = cov_chol + sp_chol[:, i]*sp_chol[:, i].transpose()
    cov_chol = 0.5*cov_chol

    return sp_chol


def main():
    global sigma_a, sigma_b
    sigma_a = 2.5
    sigma_b = 1.5
    corr = 0.0
    cov = corr*sigma_a*sigma_b
    mean = np.matrix([[0.0], [0.0]])
    A = np.matrix([[sigma_a*sigma_a, cov], [cov, sigma_b*sigma_b]])

    # sigma points via eigen decomposition and confidence ellipse
    sp_eig, center, width, height, angle = calculate_sp_eig(mean, A)

    global ell
    ell = Ellipse(xy=center, width=width, height=height, angle=angle, color="blue", alpha=0.5)

    # sigma points via cholesky decomposition
    sp_chol = calculate_sp_chol(mean, A)

    # plot and animation
    f, ax = plt.subplots(1, 1)
    plt.grid()
    ax.add_artist(ell)
    global plot_chol, plot_eig
    plot_chol, = plt.plot(sp_chol[0, :].tolist()[0], sp_chol[1, :].tolist()[0], 'bo')
    plot_eig, = plt.plot(sp_eig[0, :].tolist()[0], sp_eig[1, :].tolist()[0], 'r+')
    ax.set_aspect("equal")
    ax.set_xlim(-1.2*sigma_a, 1.2*sigma_a)
    ax.set_ylim(-1.2*sigma_b, 1.2*sigma_b)
    ani = FuncAnimation(f, update, frames=np.linspace(-99, 99, 199), interval=25, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
