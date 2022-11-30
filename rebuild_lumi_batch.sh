#!/bin/bash
MODULESCMD='module load PrgEnv-cray; module load rocm; module load craype-accel-amd-gfx90a;'
eval "${MODULESCMD}"
set -ex
if test Makefile.am -nt Makefile -o configure.ac -nt configure ; then autoreconf ; fi
./configure CXX=CC OPENMP_CXXFLAGS=-fopenmp CXXFLAGS='-O3 -std=c++17 -pipe -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -DCPU_DEFAULTS=0' --host=other
make
echo -e '#!/bin/bash'"\nenv | grep ^OMP_; ${MODULESCMD} make t;" > lumi_tmp_run.sh
chmod +x lumi_tmp_run.sh
srun --exclude=nid005079 --partition=eap --account=${SLURM_JOB_ACCOUNT} --time=10:00 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --gpus=1 lumi_tmp_run.sh
