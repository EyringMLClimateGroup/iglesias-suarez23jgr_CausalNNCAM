#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=sNN-all
#SBATCH --partition=gpu
#SBATCH --constraint=k80
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.sNN-all.%j.o
#SBATCH --error=LOG.sNN-all.%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=12:00:00


# Paths
scriptPath=`pwd`
logPath=${scriptPath}/logs

# Scripts
pyScript=NN_Creation
cfgFile=cfg_singlenns_neural_networks_pnas_1month.yml

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cd $scriptPath

logFile=`ls LOG.sNN-all.*`
cat $scriptPath/$cfgFile > $logFile

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
python ${pyScript}.py -c $cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
#mv $scriptPath/$logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"