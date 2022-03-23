#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=pT2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.pT2_%j.o
#SBATCH --error=LOG.pT2_%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=12:00:00
# -------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_dkrz/<script>.sh
# --------------------------------

# Paths
scriptPath=`pwd`
logPath=logs

# Scripts
pyScript=NN_Creation
cfgFile=CorrSingleNN/cfg_NN_Creation_rasp_pearson_tphystnd_2.yml
#cfgFile=CorrSingleNN/cfg_NN_Creation_rasp_pearson_phq_1.yml
logFile=`ls LOG.pT2_*`

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

echo "Create NNs"
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