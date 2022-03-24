#!/bin/bash
# mistral gpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=SHE-RS
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint=k80
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.SHE-RS_%j.o
#SBATCH --error=LOG.SHE-RS_%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=12:00:00
# -------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_dkrz/<script>.sh
# --------------------------------

logFile=`ls LOG.SHE-RS_*`

# Scripts
pyScript=SHERPA_RandomSearch

# Config file
cfgFile=220322_SHERPA_RandomSearch/cfg_SHERPA_RandomSearch.yml


# Paths
scriptPath=`pwd`
logPath=logs


## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cat ./nn_config/$cfgFile > $logFile

if [ ! -f ${scriptPath}/${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
    echo ""
fi

echo "SHERPA (RandomSearch) Hyperparameter Tuning"
source /pf/b/b309172/.bashrc
conda activate causalnncam
python ${pyScript}.py -c ./nn_config/$cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"