#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=sgl_tphystnd_1-5
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=LOG.sgl_tphystnd_1-5_%j.o
#SBATCH --error=LOG.sgl_tphystnd_1-5_%j.o
##SBATCH --exclusive
#SBATCH --mail-type=FAIL
#SBATCH --time=00:30:00
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
cfgFile=cfg_${pyScript}_tphystnd_1-5.yml

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

logFile=`ls LOG.sgl_tphystnd_1-5_*`
cat ./nn_config/$cfgFile > $logFile

if [ ! -f ${scriptPath}/${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
fi
echo ""

echo "Run PCMCI"
source /pf/b/b309172/.bashrc
conda activate causalnncam
python ${pyScript}.py -c ./nn_config/$cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $scriptPath/$logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"