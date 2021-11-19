#!/bin/bash
#=============================================================================
# =====================================
# mistral batch job parameters
#-----------------------------------------------------------------------------
#SBATCH --account=bd1179

#SBATCH --job-name=dT197
#SBATCH --partition=compute2,compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
# the following is needed to work around a bug that otherwise leads to
# a too low number of ranks when using compute,compute2 as queue
#SBATCH --mem=0
#SBATCH --output=LOG.NN_tb_Rasp_dT197_%j.o
#SBATCH --error=LOG.NN_tb_Rasp_dT197_%j.o
#SBATCH --exclusive
#SBATCH --time=08:00:00
#-----------------------------------------------------------------------------
# 
#=============================================================================

# Paths
scriptPath=`pwd`
logPath=logs

# Scripts
pyScript=NN_Creation
cfgFile=cfg_NN_Creation_tphystnd_197hPa.yml

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

logFile=`ls LOG.NN_tb_Rasp_dT197_*`
cat ./nn_config/tests/$cfgFile > $logFile

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
python ${pyScript}.py -c ./nn_config/tests/$cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"