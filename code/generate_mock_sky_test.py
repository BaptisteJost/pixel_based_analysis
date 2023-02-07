from generate_SCMB import get_Cl_cmbBB
import healpy as hp
import numpy as np
import time
from astropy import units as u
from bjlib.lib_project import cl_rotation
from pixel_based_angle_estimation import data_and_model_quick
from residuals import multi_freq_get_sky_fg
import IPython

nside = 64

r = 0.0
bire_angle = (0.0*u.deg).to(u.rad)
A_lens = 1

path_BB_local = '/home/baptiste/BBPipe'
path_BB_NERSC = '/global/homes/j/jost/BBPipe'
path_BB = path_BB_NERSC
# ps_planck = hp.read_cl(path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')
print('test2')

ps_planck = get_Cl_cmbBB(Alens=A_lens, r=r, path_BB=path_BB)
cmb_spectra = ps_planck
cmb_spectra = cl_rotation(ps_planck.T, bire_angle).T
start = time.time()
print('test3')

freq_number = 22
data, model_data = data_and_model_quick(
    miscal_angles_array=np.array([0]*freq_number)*u.rad, bir_angle=0*u.rad,
    frequencies_by_instrument_array=[1]*freq_number, nside=nside,
    sky_model='c1s0d0', sensitiviy_mode=0,
    one_over_f_mode=0, instrument='LiteBIRD', overwrite_freq=None)

data.get_pysm_sky()
fg_freq_maps_full = multi_freq_get_sky_fg(data.sky, data.frequencies)
print(data.frequencies)

lmin = 2
lmax = 125
ell_noise = np.linspace(2, lmax-1, lmax-2, dtype=int)

instrument_LB = np.load('data/instrument_LB_IMOv1.npy', allow_pickle=True).item()

noise_lvl = np.array([instrument_LB[f]['P_sens'] for f in instrument_LB.keys()])
beam_rad = np.array([instrument_LB[f]['beam']
                     for f in instrument_LB.keys()]) * u.arcmin.to(u.rad)
noise_nl = []
for f in range(len(noise_lvl)):
    Bl = hp.gauss_beam(beam_rad[f], lmax=lmax-1)[2:]
    Bl = np.ones(Bl.shape)
    noise = (noise_lvl[f]*np.pi/60/180)**2 * np.ones(len(ell_noise))
    noise_nl.append(noise / (Bl**2))
noise_nl = np.array(noise_nl)

for i in range(99):
    map_CMB = hp.synfast(cmb_spectra, nside, new=True)[1:]

    freq_maps = []
    for f in range(freq_number):
        map_noise = hp.synfast([noise_nl[f]*0, noise_nl[f], noise_nl[f],
                                noise_nl[f]*0], nside, new=True)[1:]
        map_f = map_CMB + fg_freq_maps_full[2*f:2*f+1] + map_noise
        freq_maps.append(map_f[0])
        freq_maps.append(map_f[1])

    freq_maps = np.array(freq_maps)
    np.save('/global/homes/j/jost/these/pixel_based_analysis/results_and_data/pipeline_data/test_mock/data/mock_LB_nobeam' +
            str(i).zfill(4)+'.npy', freq_maps)
