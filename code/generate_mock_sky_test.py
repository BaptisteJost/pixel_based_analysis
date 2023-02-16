from generate_SCMB import get_Cl_cmbBB
import healpy as hp
import numpy as np
import time
from astropy import units as u
from bjlib.lib_project import cl_rotation
from pixel_based_angle_estimation import data_and_model_quick
from residuals import multi_freq_get_sky_fg
import IPython
#from config_pipeline_copy import *

nside = 64
print('NSIDE=', nside)
r = 0.0
bire_angle = (0.0*u.deg).to(u.rad)
A_lens = 1

machine = 'idark'
if machine == 'local':
    path_BB = '/home/baptiste/BBPipe'
elif machine == 'NERSC':
    path_BB = '/global/homes/j/jost/BBPipe'
elif machine == 'idark':
    path_BB = '/home/jost/code/BBPipe'
else:
    print(machine, ' doesn\'t have BBPipe path specified')
# path_BB = path_BB_idark
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
# IPython.embed()
noise_nl = []
cmb_spectra_beamed = []
fg_maps_smoothed = []
for f in range(len(noise_lvl)):
    # Bl = hp.gauss_beam(beam_rad[f], lmax=lmax-1)[2:]
    Bl = hp.gauss_beam(beam_rad[f], lmax=3*nside)
    # Bl = np.ones(Bl.shape)
    # noise = (noise_lvl[f]*np.pi/60/180)**2 * np.ones(len(ell_noise))
    noise = (noise_lvl[f]*np.pi/60/180)**2 * np.ones(3*nside+1)

    cmb_spectra_beamed.append(cmb_spectra[:, :3*nside+1] / (Bl**2))
    '''Noise not affected by beam!'''
    # noise_nl.append(noise / (Bl**2))

    fg_maps_smoothed.append(hp.smoothing(
        np.array([fg_freq_maps_full[2*f]*0, fg_freq_maps_full[2*f],
                  fg_freq_maps_full[2*f+1]]), fwhm=beam_rad[f]))
    noise_nl.append(noise)
noise_nl = np.array(noise_nl)
cmb_spectra_beamed = np.array(cmb_spectra_beamed)
fg_maps_smoothed = np.array(fg_maps_smoothed)

if machine == 'idark':
    output_dir = '/home/jost/simu/LB_mock/fullsky_withbeam/'
elif machine == 'NERSC':
    output_dir = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/pipeline_data/test_mock/data/'
else:
    print(machine, ' doesn\'t have a specified output directory ')

print('output dir = ', output_dir)

for i in range(99):
    # map_CMB = hp.synfast(cmb_spectra, nside, new=True)[1:]
    print('generating map #',i)
    freq_maps = []
    for f in range(freq_number):
        map_noise = hp.synfast([noise_nl[f]*0, noise_nl[f], noise_nl[f],
                                noise_nl[f]*0], nside, new=True)[1:]
        map_CMB = hp.synfast(cmb_spectra_beamed[f], nside, new=True)[1:]
        # map_f = map_CMB + fg_freq_maps_full[2*f:2*f+1] + map_noise
        map_f = map_CMB + fg_maps_smoothed[f][1:] + map_noise
        freq_maps.append(map_f[0])
        freq_maps.append(map_f[1])

    freq_maps = np.array(freq_maps)
    # np.save('/global/homes/j/jost/these/pixel_based_analysis/results_and_data/pipeline_data/test_mock/data/mock_LB_nobeam' +
    #        str(i).zfill(4)+'.npy', freq_maps)
    # IPython.embed()
    np.save(output_dir + 'mock_LB' +
            str(i).zfill(4)+'.npy', freq_maps)
