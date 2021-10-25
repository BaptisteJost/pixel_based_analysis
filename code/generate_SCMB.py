import healpy as hp
import numpy as np
import time
from astropy import units as u
import bjlib.lib_project as lib
from mpi4py import MPI


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


def main():
    print('test1')
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    size_mpi = comm.Get_size()
    print(mpi_rank, size_mpi)
    root = 0

    nside = 2048
    nsim_cmb = 1000
    r = 0.0
    bire_angle = (0.35*u.deg).to(u.rad)
    A_lens = 1
    path_BB_local = '/home/baptiste/BBPipe'
    path_BB_NERSC = '/global/homes/j/jost/BBPipe'
    path_BB = path_BB_NERSC
    # ps_planck = hp.read_cl(path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')
    print('test2')

    ps_planck = get_Cl_cmbBB(Alens=A_lens, r=r, path_BB=path_BB)
    cmb_spectra = ps_planck
    cmb_spectra = lib.cl_rotation(ps_planck.T, bire_angle).T
    start = time.time()
    print('test3')

    S_sim = []
    start_loop = time.time()
    for i in range(nsim_cmb//size_mpi):
        map = hp.synfast(cmb_spectra, nside, new=True)
        S_sim.append(np.einsum('ip,jp-> ij', map[1:], map[1:]))
        del map
    S_sim = np.array(S_sim)
    end_loop = time.time()
    print('time loop = ', end_loop - start_loop)
    print('time step = ', (end_loop - start_loop)/(nsim_cmb//size_mpi))
    S_sim_mpi = None
    if comm.rank == root:
        S_sim_mpi = np.empty((np.shape(S_sim)[0]*size_mpi,)+np.shape(S_sim)[1:])
    comm.Gather(S_sim, S_sim_mpi, root)

    if comm.rank == root:
        S_cmb = np.sum(S_sim_mpi, axis=0)/((nsim_cmb//size_mpi)*size_mpi)/hp.nside2npix(nside)
        print('time=', time.time()-start)

        S_cmb_name = 'data/S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim_cmb, r, bire_angle.value).replace(
            '.', 'p') + '.npy'
        S_cmb_mpi_name = 'data/S_cmb_mpi_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim_cmb, r, bire_angle.value).replace(
            '.', 'p') + '.npy'

        np.save(S_cmb_name, S_cmb)
        np.save(S_cmb_mpi_name, S_sim_mpi)
    exit()
# IPython.embed()
# cl_list_average = np.mean(cl_list, axis=0)
# map_mean = np.mean(map_list, axis=0)
# cl_mapaverage = hp.anafast(map_mean)
######################################################


# MAIN CALL
if __name__ == "__main__":
    main()
