import IPython
import healpy as hp
import numpy as np
from astropy import units as u
import os
import time
from mpi4py import MPI


pixel_path_NERSC = '/global/u2/j/jost/these/pixel_based_analysis/code/'
pixel_path_idark = '/home/jost/code/pixel_based_analysis/code/'
pixel_path = pixel_path_idark

noise_path_NERSC = '/global/cfs/cdirs/litebird/simulations/maps/PTEP_20200915_compsep/noise/'
noise_path_idark = '/lustre/work/jost/simulations/LB_phase1/noise_maps/noise/'
noise_path = noise_path_idark
nside_output = 64
nside_input = 512  # 64 is for debug, true is 512
common_beam = 80
arcmin2rad = 1 * u.arcmin.to(u.rad)
Bl_gauss_common = hp.gauss_beam(common_beam*arcmin2rad, lmax=3*nside_input, pol=True)[:, 1]

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank_mpi = comm.rank  # rank=0
# rank = 99
print('MPI size = ', size)
print('MPI rank = ', rank_mpi)

instrument_LB = np.load(pixel_path+'data/instrument_LB_IMOv1.npy',
                        allow_pickle=True).item()
scaling_factor_for_sensitivity_due_to_transfer_function = np.array([1.03538701, 1.49483298, 1.68544978, 1.87216344, 1.77112946, 1.94462893,
                                                                    1.83405669, 1.99590729, 1.87252192, 2.02839885, 2.06834379, 2.09336487,
                                                                    1.93022977, 1.98931453, 2.02184001, 2.0436672,  2.05440473, 2.04784698,
                                                                    2.08458758, 2.10468234, 2.1148482, 2.13539999])


nsim_tot = 100
print('simulation per process', nsim_tot//size)
print('leftover simulation', nsim_tot % size)
if nsim_tot % size != 0:
    print('WARNING: some simulations won\'t be taken into account!')


start_total_loop = time.time()
std_list = []
for iter in range(nsim_tot//size):
    sim_num = rank_mpi * (nsim_tot//size) + iter
    print('ITER #', iter)
    print('sim_num =', sim_num)
    std_list_freq = []
    freq_counter = 0
    start_freq_loop = time.time()

    '''noise cov could be done in a spearate loop, it is overwritten at ever step here...'''
    noise_cov_rescale_sqrt = []
    noise_cov_sqrt = []
    for freq_tag in instrument_LB.keys():
        print(freq_tag)
        sens_rescale = instrument_LB[freq_tag]['P_sens'] / \
            scaling_factor_for_sensitivity_due_to_transfer_function[freq_counter]
        noise_cov_rescale_sqrt.append((sens_rescale / hp.nside2resol(nside_output, arcmin=True)))
        noise_cov_sqrt.append((instrument_LB[freq_tag]['P_sens'] /
                               hp.nside2resol(nside_output, arcmin=True)))

        Bl_gauss_fwhm = hp.gauss_beam(
            instrument_LB[freq_tag]['beam']*arcmin2rad, lmax=3*nside_input, pol=True)[:, 1]

        path_noise_map = noise_path + \
            str(sim_num).zfill(4)+'/'+freq_tag+'_noise_FULL_' + \
            str(sim_num).zfill(4)+'_PTEP_20200915_compsep.fits'
        print(path_noise_map)
        if not os.path.exists(path_noise_map):
            print('PATH DOESN\'T EXIST')
            std_list_freq.append(None)
            continue
        noise_map_totfield = hp.read_map(path_noise_map, field=(0, 1, 2))
        alms = hp.map2alm(noise_map_totfield, lmax=3*nside_input)
        alms_beamed = []
        for alm_ in alms:
            alms_beamed.append(hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=False))
        output_map = hp.alm2map(alms_beamed, nside_output)
        std_list_freq.append(np.std(output_map))
        freq_counter += 1
    noise_cov_rescale_sqrt = np.array(noise_cov_rescale_sqrt)
    noise_cov_sqrt = np.array(noise_cov_sqrt)
    std_list.append(std_list_freq)
    print('time in freq loop=', time.time() - start_freq_loop)
    print('average time for one freq loop=', (time.time() - start_freq_loop)/freq_counter)

print('time in global loop=', time.time() - start_total_loop)


std_list = np.array(std_list)
print('nsim_tot//size = ', nsim_tot//size)
print('iter+1 = ', iter+1)
print('defined recbuff shape:', size, nsim_tot//size, freq_counter)
# IPython.embed()
recvbuf = None
if rank_mpi == 0:
    recvbuf = np.empty([size, iter+1, freq_counter], dtype='d')

comm.Gather(std_list, recvbuf, root=0)

if rank_mpi == 0:
    # TODO: attention au differentes frequences
    print(np.mean(recvbuf, axis=(0, 1))/noise_cov_rescale_sqrt)
    np.save('results_std_noise_beam.npy', recvbuf)
    np.save('results_meanstd_beam_over_rescalednoisecov',
            np.mean(recvbuf, axis=(0, 1))/noise_cov_rescale_sqrt)
    np.save('results_meanstd_beam_over_noisecov', np.mean(recvbuf, axis=(0, 1))/noise_cov_sqrt)
