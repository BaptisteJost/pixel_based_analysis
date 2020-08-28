import matplotlib.pyplot as plt
import time
import IPython
import numpy as np
from scipy.optimize import minimize
import bjlib.likelihood_SO as lSO


def main():

    data = lSO.sky_map()
    model = lSO.sky_map()

    data.from_pysm2data()
    model.from_pysm2data()

    data.get_noise()
    model.get_noise()
    # IPython.embed()
    data.get_projection_op()
    model.get_projection_op()
    # IPython.embed()

    data.data2alm()
    model.data2alm()

    data.get_primordial_spectra()
    model.get_primordial_spectra()

    start = time.time()
    grid = np.arange(-1*np.pi, 1*np.pi, 2*np.pi/100)
    # for i in grid:
    #     get_chi_squared([i], data, model)

    # print('time chi2 in s = ', time.time() - start)
    # IPython.embed()

    start = time.time()
    prior_ = False
    results = minimize(lSO.get_chi_squared, [0, 0, 0], (data, model, prior_),
                       bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])
    print('time minimize in s = ', time.time() - start)

    IPython.embed()
    # print('results = ', results.x)
    # IPython.embed()
    # print('hessian = ', results.hess_inv)

    # visu.corner_norm(results.x, results.hess_inv)
    plt.show()
    # bir_grid, misc_grid = np.meshgrid(grid, grid,
    # indexing='ij')
    start = time.time()
    lSO.get_chi_squared([0, 0, 0], data, model, prior=True)
    print('time chi2 prior = ', time.time() - start)
    # slice_chi2 = np.array([[get_chi_squared([i, j, 0, 0], data, model) for i in grid]
    #                        for j in grid])
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(grid, misc_grid, slice_chi2, cmap=cm.viridis)
    # plt.show()

    plt.plot(grid, [-lSO.get_chi_squared([i], data, model) for i in grid])
    print('time grid in s = ', time.time() - start)

    # plt.yscale('log')
    plt.show()

    # IPython.embed()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
