#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=sherpa
#SBATCH --partition=gpu
#SBATCH --constraint=k80
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.sherpa.%j.o
#SBATCH --error=LOG.sherpa.%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=12:00:00


# Paths
scriptPath=`pwd`
logPath=${scriptPath}/logs

# Scripts
pyScript=sherpa_hyperparameter_tuning
cfgFile="nn_config/cfg_sherpa.yml"

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cd $scriptPath

logFile=`ls LOG.sherpa.*`
cat $scriptPath/$cfgFile > $logFile

if [ ! -f ${scriptPath}/${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
    echo ""
fi

echo "Sherpa hyperparameter tuning"
source /pf/b/b309172/.bashrc
conda activate causalnncam
python ${pyScript}.py -c $cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
#mv $scriptPath/$logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"