import numpy as np
from astropy import units as u
import bjlib.likelihood_SO as lSO


freq_number = 7
fsky = 0.6
lmin = 51
lmax = 1500
nside = 2048


sky_model = 'c1s0d0'
sensitiviy_mode = 0
one_over_f_mode = 0
A_lens_true = 1
instrument = 'Planck'
freq_by_instru = [1]*freq_number
beta_true = (0.0 * u.deg).to(u.rad)
true_miscal_angles = np.array([0]*freq_number)*u.rad


data = lSO.sky_map(bir_angle=beta_true, miscal_angles=true_miscal_angles,
                   frequencies_by_instrument=freq_by_instru,
                   nside=nside, sky_model=sky_model, instrument=instrument)
data.get_pysm_sky()
data.get_frequency()
data.get_freq_maps()

np.save('data/pysm_cmbsky_c1_2048.npy', data.cmb_freq_maps)
np.save('data/pysm_dustsky_d0_2048.npy', data.dust_freq_maps)
np.save('data/pysm_synchsky_s0_2048.npy', data.sync_freq_maps)
