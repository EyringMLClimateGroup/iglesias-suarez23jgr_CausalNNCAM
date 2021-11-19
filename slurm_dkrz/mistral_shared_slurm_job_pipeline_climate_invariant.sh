#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=ClInv
#SBATCH --partition=shared
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --output=LOG.ClInv_pipeline_%j.o
#SBATCH --error=LOG.ClInv_pipeline_%j.o
##SBATCH --exclusive
#SBATCH --mail-type=FAIL
#SBATCH --time=7-00:00:00
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
cfgFile=cfg_${pyScript}_climate_invariant.yml

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

logFile=`ls LOG.ClInv_pipeline_*`
cat ./nn_config/$cfgFile > $logFile

if [ ! -f ${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
fi
echo ""

echo "Run PCMCI for Climate Invariant"
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