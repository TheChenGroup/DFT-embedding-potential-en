#!/bin/bash
#SBATCH --job-name=292-a-DMFET-0.02
#SBATCH --output=test.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=36
#SBATCH --time=71:00:00
#SBATCH -p regular

module load compiler/intel/intel-compiler-2019u3
module load mpi/intelmpi/2019u3
module load apps/vasp/5.4.4/intelmpi
export PYSCF_TMPDIR=/public/home/yilinchen/tmpt

for pyscf_file in ./*.py
do
    ~/anaconda2/bin/python ${pyscf_file} > ${pyscf_file%.*}.out

done

