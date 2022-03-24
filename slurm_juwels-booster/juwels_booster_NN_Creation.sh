#!/bin/bash
# juwels-booster batch job parameters
# --------------------------------
#SBATCH --account=cesmtst
#SBATCH --job-name=p59TQ3
#SBATCH --partition=booster
####SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=LOG.p59TQ3-out_%j
#SBATCH --error=LOG.p59TQ3-out_%j
#SBATCH --mail-type=FAIL
#SBATCH --time=08:00:00
# --------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_juwels/<script>.sh
# --------------------------------

logFile=`ls LOG.p59TQ3-out_*`

# Scripts
pyScript=NN_Creation
#cfgFile=220224_threshold_optimization/thrs/cfg_NN_Creation_rasp_parcorr_thrs-0.05_tphystnd_29.yml
#cfgFile=220224_threshold_optimization/thrs/cfg_NN_Creation_rasp_parcorr_thrs-0.05_phq_29.yml

#cfgFile=220224_threshold_optimization/pdf/cfg_NN_Creation_rasp_parcorr_pdf-0.59_tphystnd_24.yml
#cfgFile=220224_threshold_optimization/pdf/cfg_NN_Creation_rasp_parcorr_pdf-0.59_phq_24.yml

#cfgFile=220224_threshold_optimization/single/cfg_NN_Creation_rasp_tphystnd_28.yml
#cfgFile=220224_threshold_optimization/single/cfg_NN_Creation_rasp_phq_28.yml

# e.g., job-name=p59TQ21
cfgFile=220308_rasp_etal_2018_optimized_threshold/quantile-0.59/cfg_NN_Creation_rasp_parcorr_pdf-0.59_phq-tphystnd_3.yml

# e.g., job-name=sglTQ29
#cfgFile=220308_rasp_etal_2018_optimized_threshold/single/cfg_NN_Creation_rasp_phq-tphystnd_6.yml

# Test; e.g., job-name=test
#cfgFile=220310_distributed_epoch_test/cfg_testing.yml


# Paths
scriptPath=`pwd`
logPath=logs


## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cat ./nn_config/$cfgFile > $logFile

echo "Create NNs as per Rasp et al. (2018)."
source /p/home/jusers/iglesias-suarez1/juwels/.bash_profile
conda activate causalnncam

if [ ! -f ${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    jupyter nbconvert --to script ${pyScript}.ipynb
fi
echo ""

python ${pyScript}.py -c ./nn_config/$cfgFile

# Clean-up
rm ${pyScript}.py
if [ ! -d $logPath ]; then
    mkdir -p $logPath
fi
mv $logFile $logPath/$logFile

echo ""
echo "---------- Finished $0 ----------"
