# import IPython
import healpy as hp
import numpy as np
import time
from residuals import get_Cl_cmbBB
from astropy import units as u
import bjlib.lib_project as lib


nside = 128
nsim_cmb = 1000
r = 0.0
bire_angle = 0.01 * u.rad
A_lens = 1
path_BB_local = '/home/baptiste/BBPipe'
path_BB_NERSC = '/global/homes/j/jost/BBPipe'
path_BB = path_BB_local
# ps_planck = hp.read_cl(path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')


ps_planck = get_Cl_cmbBB(Alens=A_lens, r=r, path_BB=path_BB)
cmb_spectra = ps_planck
cmb_spectra = lib.cl_rotation(ps_planck.T, bire_angle).T
start = time.time()
map_list2 = []
for i in range(nsim_cmb):
    map_list2.append(hp.synfast(cmb_spectra, nside, new=True))
map_list = np.array(map_list2)
print('time=', time.time()-start)
map_array_reshaped = np.swapaxes(map_list, axis1=0, axis2=1)
cl_list = [hp.anafast(map_list[i]) for i in range(nsim_cmb)]

S_cmb = np.einsum(
    'isp,jsp->ij', map_array_reshaped[1:], map_array_reshaped[1:])/nsim_cmb/hp.nside2npix(nside)
print('total time=', time.time()-start)

S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim_cmb, r, bire_angle.value).replace(
    '.', 'p') + '.npy'

np.save(S_cmb_name, S_cmb)
# IPython.embed()
# cl_list_average = np.mean(cl_list, axis=0)
# map_mean = np.mean(map_list, axis=0)
# cl_mapaverage = hp.anafast(map_mean)
