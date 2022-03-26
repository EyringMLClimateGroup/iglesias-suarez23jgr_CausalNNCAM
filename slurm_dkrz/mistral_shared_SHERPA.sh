#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#####SBATCH --account=bd1083
#SBATCH --job-name=SHE-GS
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=LOG.SHE-GS_%j.o
#SBATCH --error=LOG.SHE-GS_%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=7-00:00:00
# -------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_dkrz/<script>.sh
# --------------------------------

logFile=`ls LOG.SHE-GS_*`

# Scripts
#pyScript=SHERPA_RandomSearch
pyScript=SHERPA_GridSearch
#pyScript=SHERPA_ASHA

# Config file
#cfgFile=220322_SHERPA_RandomSearch/cfg_SHERPA_RandomSearch.yml
cfgFile=220324_SHERPA_GridSearch/cfg_SHERPA_GridSearch.yml
#cfgFile=220322_SHERPA_ASHA/cfg_SHERPA_ASHA.yml


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

echo "SHERPA Hyperparameter Tuning"
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