import shutil
import os
import argparse
from datetime import date
import click


def write_bash_spec(job_id, path):
    spec_batch_name = path+"batch_spec_"+job_id+".sh"
    fout = open(spec_batch_name, "wt")
    fout.write("#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -C haswell\n\
#SBATCH -q debug\n\
#SBATCH -J Sp"+job_id+"\n\
#SBATCH --mail-user=jost@apc.in2p3.fr\n\
#SBATCH --mail-type=ALL\n\
#SBATCH -t 00:30:00\n\
#OpenMP settings:\n\
export OMP_NUM_THREADS=1\n\
export OMP_PLACES=threads\n\
export OMP_PROC_BIND=spread\n\
\n\
\n\
#run the application:\n\
conda activate myenv\n\
srun -n 1 -c 64 --cpu_bind=core python "+path+"double_sample_spec_copy.py " + path)

    fout.close()
    return spec_batch_name


def write_bash_cosmo(job_id, path):
    cosmo_batch_name = path+"batch_cosmo_MCMC_"+job_id+".sh"
    fout = open(cosmo_batch_name, "wt")
    fout.write("#!/bin/bash\n\
#SBATCH -N 10\n\
#SBATCH -C haswell\n\
#SBATCH -q regular\n\
#SBATCH -J Co"+job_id+"\n\
#SBATCH --mail-user=jost@apc.in2p3.fr\n\
#SBATCH --mail-type=ALL\n\
#SBATCH -t 02:00:00\n\
#OpenMP settings:\n\
export OMP_NUM_THREADS=1\n\
export OMP_PLACES=threads\n\
export OMP_PROC_BIND=spread\n\
\n\
\n\
#run the application:\n\
conda activate myenv\n\
srun -n 640 -c 1 --cpu_bind=threads python "+path+"double_sample_cosmo_copy.py " + path)

    fout.close()
    return cosmo_batch_name
# mkdir /tmp/jost_dir_"+job_id+"/\n\
# sbcast "+path+"double_sample_cosmo_copy.py /tmp/jost_dir_"+job_id+"/jost_double_sample_cosmo_copy.py\n\
# sbcast "+path+"config_copy.py /tmp/jost_dir_"+job_id+"/config_copy.py\n\
# if [ ! -d /tmp/jost_dir_"+job_id+"/ ] \n\
# then\n\
#     echo \"creating directory\"\n\
#     mkdir /tmp/jost_dir_"+job_id+"/\n\
# fi \n\


def write_bash_cosmo_min(job_id, path):
    cosmo_batch_name = path+"batch_cosmo_min_"+job_id+".sh"
    fout = open(cosmo_batch_name, "wt")
    fout.write("#!/bin/bash\n\
#SBATCH -N 10\n\
#SBATCH -C haswell\n\
#SBATCH -q debug\n\
#SBATCH --mail-user=jost@apc.in2p3.fr\n\
#SBATCH --mail-type=ALL\n\
#SBATCH -t 00:30:00\n\
#OpenMP settings:\n\
export OMP_NUM_THREADS=1\n\
export OMP_PLACES=threads\n\
export OMP_PROC_BIND=spread\n\
\n\
\n\
#run the application:\n\
conda activate myenv\n\
srun -n 640 -c 1 --cpu_bind=threads python "+path+"double_sample_cosmo_min_copy.py " + path)

    fout.close()
    return cosmo_batch_name


def write_bash_plot(job_id, path):
    plot_batch_name = path+"batch_plot_"+job_id+".sh"
    fout = open(plot_batch_name, "wt")
    fout.write("#!/bin/bash\n\
#SBATCH -N 1\n\
#SBATCH -C haswell\n\
#SBATCH -q debug\n\
#SBATCH --mail-user=jost@apc.in2p3.fr\n\
#SBATCH --mail-type=ALL\n\
#SBATCH -t 00:30:00\n\
#OpenMP settings:\n\
export OMP_NUM_THREADS=1\n\
export OMP_PLACES=threads\n\
export OMP_PROC_BIND=spread\n\
\n\
\n\
#run the application:\n\
conda activate myenv\n\
srun -n 1 -c 64 --cpu_bind=cores python "+path+"plot_double_sampling_copy.py " + path)

    fout.close()
    return plot_batch_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id", help="job id")
    parser.add_argument("--job_date", help="job date, default today",
                        default=date.today().strftime('%Y%m%d'))
    # parser.add_argument("--precision_index", help="prior precision in deg !!")
    args = parser.parse_args()

    pixel_path = '/global/u2/j/jost/these/pixel_based_analysis/'
    save_path_ = pixel_path + 'results_and_data/full_pipeline/prior_grid/' + \
        args.job_date

    save_path = save_path_ + '_' + args.job_id + '/'
    print()
    print(save_path)
    print(args.job_id)
    if os.path.exists(save_path):
        print('Path already exists, config file will NOT be updated but other scripts will be.')
        if click.confirm('Do you want to continue?'):
            print('Continue...')
        else:
            print('Abort...')
            exit()
    else:
        os.mkdir(save_path)
        shutil.copy('config.py', save_path+'config_copy.py')
    shutil.copy('double_sample_spec.py', save_path+'double_sample_spec_copy.py')
    shutil.copy('double_sample_cosmo.py', save_path+'double_sample_cosmo_copy.py')
    shutil.copy('double_sample_cosmo_min.py', save_path+'double_sample_cosmo_min_copy.py')
    shutil.copy('plot_double_sampling.py', save_path+'plot_double_sampling_copy.py')

    print('Spectral samples...')
    if os.path.exists(save_path+'spec_samples.npy'):
        print('spectral samples already computed')
    else:
        spec_batch_name = write_bash_spec(args.job_id, save_path)
        print('Sampling spectral likelihood...')
        # os.system('sbatch -W ' + spec_batch_name)

    print('Cosmo samples...')
    if os.path.exists(save_path+'cosmo_samples_MCMC.npy'):
        print('cosmo samples MCMC already computed')
    else:
        cosmo_batch_MCMC = write_bash_cosmo(args.job_id, save_path)
        print('Sampling cosmo likelihood MCMC...')
        print('bash file : ', cosmo_batch_MCMC)
        # os.system('sbatch ' + cosmo_batch_MCMC)

    if os.path.exists(save_path+'cosmo_samples_min.npy'):
        print('cosmo samples MIN already computed')
    else:
        cosmo_batch_min = write_bash_cosmo_min(args.job_id, save_path)
        print('Sampling cosmo likelihood MIN...')
        print('bash file : ', cosmo_batch_min)

        # os.system('sbatch ' + cosmo_batch_min)

    print('Plotting...')
    plot_batch_name = write_bash_plot(args.job_id, save_path)
    print('bash file plot : ', plot_batch_name)

    # os.system('sbatch ' + plot_batch_name)

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
