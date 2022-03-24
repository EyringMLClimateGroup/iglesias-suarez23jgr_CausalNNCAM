#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=s28T
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint=k80
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.s28T_%j.o
#SBATCH --error=LOG.s28T_%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=12:00:00
# -------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_dkrz/<script>.sh
# --------------------------------

logFile=`ls LOG.s28T_*`

# Scripts
pyScript=NN_Creation
#cfgFile=220224_threshold_optimization/thrs/cfg_NN_Creation_rasp_parcorr_thrs-0.05_tphystnd_29.yml
#cfgFile=220224_threshold_optimization/thrs/cfg_NN_Creation_rasp_parcorr_thrs-0.05_phq_29.yml

#cfgFile=220224_threshold_optimization/pdf/cfg_NN_Creation_rasp_parcorr_pdf-0.59_tphystnd_24.yml
#cfgFile=220224_threshold_optimization/pdf/cfg_NN_Creation_rasp_parcorr_pdf-0.59_phq_24.yml

cfgFile=220224_threshold_optimization/single/cfg_NN_Creation_rasp_tphystnd_28.yml
#cfgFile=220224_threshold_optimization/single/cfg_NN_Creation_rasp_phq_28.yml


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