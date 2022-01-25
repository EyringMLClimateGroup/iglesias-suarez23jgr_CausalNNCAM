#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=srun
#SBATCH --partition=compute
#SBATCH --nodes=3
#SBATCH --constraint=256G
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.nsrun-out_%j
#SBATCH --error=LOG.nsrun-out_%j
#SBATCH --mail-type=FAIL
#SBATCH --time=01:00:00
# --------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_dkrz/<script>.sh
# --------------------------------

# Paths
scriptPath=`pwd`
logPath=logs


# Scripts
pyScript=pipeline


# Variables & Levels
variables_3D=" tphystnd phq "
variables_2D=" fsns flns fsnt flnt prect "
levels=" 3.643 7.595 14.357 24.612 38.268 54.595 72.012 87.821 103.317 121.547 142.994 168.225 197.908 232.829 273.911 322.242 379.101 445.993 524.687 609.779 691.389 763.404 820.858 859.535 887.020 912.645 936.198 957.485 976.325 992.556 "


## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

logFile=`ls LOG.nsrun-out_*`
#cat ./nn_config/$cfgFile > $logFile
cat ./slurm_dkrz/tests/test/$cfgFile > $logFile

if [ ! -f ${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
fi
echo ""

echo "Run PCMCI for Rasp et al."
source /pf/b/b309172/.bashrc
conda activate causalnncam

count=1
for iVar in $variables_3D; do
    for iLev in $levels; do
	cfgFile=cfg_pipeline_${iVar}_${iLev}.yml
	if test -f "slurm_dkrz/tests/test/$cfgFile"; then
	    echo "$cfgFile exists."
	else
	    echo "create ${cfgFile}..."
	    cp slurm_dkrz/cfg_pipeline.yml slurm_dkrz/tests/test/$cfgFile
	    sed -i -- "s/variable/${iVar}/g" slurm_dkrz/tests/test/$cfgFile
	    sed -i -- "s/level_to_be_analysed/[${iLev}]/g" slurm_dkrz/tests/test/$cfgFile
	fi
	echo $count
	srun -N 1 --ntasks-per-node=24 -c 1 python ${pyScript}.py -c ./slurm_dkrz/tests/test/$cfgFile &
	let count++
    done
done

for iVar in $variables_2D; do
    cfgFile=cfg_pipeline_${iVar}_False.yml
    if test -f "slurm_dkrz/tests/test/$cfgFile"; then
	echo "$cfgFile exists."
    else
	echo "create ${cfgFile}..."
	cp slurm_dkrz/cfg_pipeline.yml slurm_dkrz/tests/test/$cfgFile
	sed -i -- "s/variable/${iVar}/g" slurm_dkrz/tests/test/$cfgFile
	sed -i -- "s/level_to_be_analysed/False/g" slurm_dkrz/tests/test/$cfgFile
    fi
    srun -N 1 --ntasks-per-node=24 -c 1 python ${pyScript}.py -c ./slurm_dkrz/tests/test/$cfgFile &
    echo $count
    let count++
done

wait

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $logFile $logPath/$logFile


echo ""
echo "---------- Finished $0 ----------"