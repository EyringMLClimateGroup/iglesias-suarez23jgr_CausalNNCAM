#!/bin/bash
# juwels-booster batch job parameters
# --------------------------------
#SBATCH --account=cesmtst
#SBATCH --job-name=dist-test
#####SBATCH --partition=booster
#SBATCH --partition=develbooster
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=LOG.dist-test-out_%j
#SBATCH --error=LOG.dist-test-out_%j
#SBATCH --mail-type=FAIL
#SBATCH --time=02:00:00
# --------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_juwels/<script>.sh
# --------------------------------

logFile=`ls LOG.dist-test-out_*`

# Scripts
pyScript=NN_Creation_distributed_training

## Config files
#
# Test; e.g., job-name=test
cfgFile=220316_distributed_training_test/cfg_NN_Creation_rasp_phq-tphystnd_12.yml
#cfgFile=220316_distributed_training_test/cfg_NN_Creation_rasp_phq-tphystnd_29.yml

#cfgFile=220224_threshold_optimization/thrs/cfg_NN_Creation_rasp_parcorr_thrs-0.05_tphystnd_29.yml
#cfgFile=220224_threshold_optimization/thrs/cfg_NN_Creation_rasp_parcorr_thrs-0.05_phq_29.yml

#cfgFile=220224_threshold_optimization/pdf/cfg_NN_Creation_rasp_parcorr_pdf-0.59_tphystnd_24.yml
#cfgFile=220224_threshold_optimization/pdf/cfg_NN_Creation_rasp_parcorr_pdf-0.59_phq_24.yml

#cfgFile=220224_threshold_optimization/single/cfg_NN_Creation_rasp_tphystnd_28.yml
#cfgFile=220224_threshold_optimization/single/cfg_NN_Creation_rasp_phq_28.yml

# e.g., job-name=p58TQ21
#cfgFile=220308_rasp_etal_2018_optimized_threshold/quantile-0.58/cfg_NN_Creation_rasp_parcorr_pdf-0.58_phq-tphystnd_21.yml#

# e.g., job-name=sglTQ29
#cfgFile=220308_rasp_etal_2018_optimized_threshold/single/cfg_NN_Creation_rasp_phq-tphystnd_6.yml

# Paths
scriptPath=`pwd`
logPath=logs


## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cat ./nn_config/$cfgFile > $logFile

echo "Create NNs via parallel distributed training."
source /p/home/jusers/iglesias-suarez1/juwels/.bash_profile
conda activate causalnncam
export PYTHONPATH="${PYTHONPATH}:/p/project/cesmtst/iglesias-suarez1/miniconda/envs/causalnncam/lib/python3.8/site-packages"

if [ ! -f ${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    jupyter nbconvert --to script ${pyScript}.ipynb
fi
echo ""

# Load the required modules
module load Stages/2020
module load GCC/9.3.0
module load OpenMPI/4.1.0rc1
module load TensorFlow/2.3.1-Python-3.8.5
module load Horovod/0.20.3-Python-3.8.5

# Make all GPUs visible per node
export CUDA_VISIBLE_DEVICES=0,1,2,3

srun python -u ${pyScript}.py -c ./nn_config/$cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"
