import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

grid1 = np.load('grid_prior_precision.npy')
grid2 = np.load('grid_prior_precision2.npy')
grid3 = np.load('grid_prior_precision3.npy')

cosmo1 = np.load('cosmo_results_grid_prior2to3.npy')
cosmo2 = np.load('cosmo_results_grid_prior2to32.npy')
cosmo3 = np.load('cosmo_results_grid_prior2to33.npy')

fisher1 = np.load('fisher_matrix_grid_prior2to3.npy')
fisher2 = np.load('fisher_matrix_grid_prior2to32.npy')
fisher3 = np.load('fisher_matrix_grid_prior2to33.npy')

spectral1 = np.load('spectral_results_grid_prior2to3.npy')
spectral2 = np.load('spectral_results_grid_prior2to32.npy')
spectral3 = np.load('spectral_results_grid_prior2to33.npy')

grid_total = np.append(np.append(grid1, grid2[1:], axis=0), grid3[1:], axis=0)
cosmo_total = np.append(np.append(cosmo1, cosmo2[1:], axis=0), cosmo3[1:], axis=0)
fisher_total = np.append(np.append(fisher1, fisher2[1:], axis=0), fisher3[1:], axis=0)
spectral_total = np.append(np.append(spectral1, spectral2[1:], axis=0), spectral3[1:], axis=0)

beta_true = (0.35*u.deg).to(u.rad)
r_true = 0.01

sigma_beta = ((1/np.sqrt(fisher_total[:, 1, 1]))*u.rad).to(u.deg)
bias_beta = ((cosmo_total[:, 1]-beta_true.value)*u.rad).to(u.deg)

sigma_r = 1/np.sqrt(fisher_total[:, 0, 0])
bias_r = cosmo_total[:, 0]-r_true


plt.plot(grid_total, sigma_beta, label=r'$\sigma (\beta_b)$')
plt.scatter(grid_total, bias_beta, label=r'$\bar{\beta}_b - \hat{\beta}_b$', marker='x')
plt.xlabel('prior precision in deg')
plt.ylabel('deg')
plt.title(r'Effect on $\beta_b$ of prior precision on 93GHz channel')
plt.legend()
plt.savefig('prior_beta')
plt.close()

plt.plot(grid_total, sigma_r, label=r'$\sigma (r)$')
plt.scatter(grid_total, bias_r, label=r'$\bar{r} - \hat{r}$', marker='x')
plt.xlabel('prior precision in deg')
plt.title('Effect on r of prior precision on 93GHz channel only')
plt.legend()
plt.savefig('prior_r')
plt.close()

exit()
