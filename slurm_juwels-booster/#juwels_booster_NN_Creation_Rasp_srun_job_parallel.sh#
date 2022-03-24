#!/bin/bash
# juwels-booster batch job parameters
# --------------------------------
#SBATCH --account=cesmtst
#SBATCH --job-name=SglTQ24
####SBATCH --partition=booster
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --output=LOG.SglTQ24-out_%j
#SBATCH --error=LOG.SglTQ24-out_%j
#SBATCH --mail-type=FAIL
#SBATCH --time=02:00:00
# --------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_juwels/<script>.sh
# --------------------------------

logFile=`ls LOG.SglTQ24-out_*`

# Variables & Levels
#variables_3D=" tphystnd "
#variables_3D=" phq "
variables_3D=" tphystnd phq "
#variables_3D=""
#levels=" 3.64 7.59 14.36 24.61 38.27 54.6 72.01 87.82 103.32 121.55 142.99 168.23 197.91 232.83 273.91 322.24 379.1 445.99 524.69 609.78 691.39 763.4 820.86 859.53 887.02 912.64 936.2 957.49 976.33 992.56 "
levels=" 887.02 "

#variables_2D=" flnt prect "
#variables_2D=" fsns flns fsnt flnt prect "
variables_2D=""


############################################################
############################################################


# Paths
scriptPath=`pwd`
logPath=logs
cfgPath=nn_config
cfgPath_parallel=$cfgPath/srun_parallel


# Scripts
pyScript=NN_Creation


## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cfgFile_tmp=cfg_NN_Creation_rasp_srun_parallel.yml

echo "Create NNs as per Rasp et al. (2018)."
source /p/home/jusers/iglesias-suarez1/juwels/.bash_profile
conda activate causalnncam

if [ ! -f ${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    jupyter nbconvert --to script ${pyScript}.ipynb
fi
echo ""

count=1
if [ -n "$variables_3D" ]; then
    for iVar in $variables_3D; do
	for iLev in $levels; do
	    cfgFile=cfg_NN_Creation_rasp_${iVar}_${iLev}.yml
	    if test -f "./${cfgPath_parallel}/$cfgFile"; then
		echo "$cfgFile exists"
	    else
		echo "create ${cfgFile}..."
		cp ./$cfgPath/$cfgFile_tmp ./$cfgPath_parallel/$cfgFile
		sed -i -- "s/variable/${iVar}/g" ./$cfgPath_parallel/$cfgFile
		sed -i -- "s/level_to_be_analysed/[${iLev}]/g" ./$cfgPath_parallel/$cfgFile
	    fi
	    echo $count
#	cat ./$cfgPath_parallel/$cfgFile > $logFile
	    srun --exclusive -N 1 --ntasks-per-node=1 python ${pyScript}.py -c ./$cfgPath_parallel/$cfgFile &
	    let count++
	done
    done  
fi

if [ -n "$variables_2D" ]; then
    for iVar in $variables_2D; do
	cfgFile=cfg_NN_Creation_rasp_${iVar}_False.yml
	if test -f "./${cfgPath_parallel}/$cfgFile"; then
	    echo "$cfgFile exists."
	else
	    echo "create ${cfgFile}..."
	    cp ./$cfgPath/$cfgFile_tmp ./$cfgPath_parallel/$cfgFile
	    sed -i -- "s/variable/${iVar}/g" ./$cfgPath_parallel/$cfgFile
	    sed -i -- "s/level_to_be_analysed/False/g" ./$cfgPath_parallel/$cfgFile
	fi
	echo $count
#	cat ./$cfgPath_parallel/$cfgFile > $logFile
	srun --exclusive -N 1 --ntasks-per-node=1 python ${pyScript}.py -c ./$cfgPath_parallel/$cfgFile &
	let count++
    done  
fi

wait

# Clean-up
rm ${pyScript}.py
rm ./$cfgPath_parallel/cfg_NN_Creation_rasp_*.yml
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $logFile $logPath/$logFile


echo ""
echo "---------- Finished $0 ----------"
