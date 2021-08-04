import IPython
import healpy as hp
import numpy as np
import time
# from residuals import get_Cl_cmbBB
from astropy import units as u
import bjlib.lib_project as lib


def get_Cl_cmbBB(Alens=1., r=0.001, path_BB='.'):
    power_spectrum = hp.read_cl(
        path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')[:, :4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r *\
            hp.read_cl(
                path_BB + '/test_mapbased_param/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:, :4000]
    return power_spectrum


nside = 512
nsim_cmb = 1000
r = 0.0
bire_angle = (0.35*u.deg).to(u.rad)
A_lens = 1
path_BB_local = '/home/baptiste/BBPipe'
path_BB_NERSC = '/global/homes/j/jost/BBPipe'
path_BB = path_BB_NERSC
# ps_planck = hp.read_cl(path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')


ps_planck = get_Cl_cmbBB(Alens=A_lens, r=r, path_BB=path_BB)
cmb_spectra = ps_planck
cmb_spectra = lib.cl_rotation(ps_planck.T, bire_angle).T
start = time.time()

S_sim = []
for i in range(nsim_cmb):
    map = hp.synfast(cmb_spectra, nside, new=True)
    S_sim.append(np.einsum('ip,jp-> ij', map[1:], map[1:]))
    del map
S_cmb = np.sum(S_sim, axis=0)/nsim_cmb/hp.nside2npix(nside)
print('time=', time.time()-start)

S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim_cmb, r, bire_angle.value).replace(
    '.', 'p') + '.npy'

np.save(S_cmb_name, S_cmb)
# IPython.embed()
# cl_list_average = np.mean(cl_list, axis=0)
# map_mean = np.mean(map_list, axis=0)
# cl_mapaverage = hp.anafast(map_mean)
