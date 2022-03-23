#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
######SBATCH --account=bd1179
#SBATCH --account=bd1083
#SBATCH --job-name=pc1-90s
#SBATCH --partition=compute,compute2,prepost
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=LOG.pc1-m4k-90s_%j.o
#SBATCH --error=LOG.pc1-m4k-90s_%j.o
#SBATCH --mail-type=FAIL
#SBATCH --time=08:00:00
# --------------------------------
# Run the script.
# From the main repository directory:
#    $> sbatch slurm_dkrz/<script>.sh
# --------------------------------

logFile=`ls LOG.pc1-m4k-90s_*`

# Scripts
pyScript=pipeline
#cfgFile=cfg_${pyScript}.yml
cfgFile=220304_pipeline_4Ks/cfg_${pyScript}_90s.yml
#cfgFile=220304_pipeline_4Ks/cfg_${pyScript}.yml

# Paths
scriptPath=`pwd`
logPath=logs



## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cat ./nn_config/$cfgFile > $logFile

if [ ! -f ${pyScript}.py ]; then
    echo "Convert jupyter notebook into python script"
    module load python3/unstable
    jupyter nbconvert --to script ${pyScript}.ipynb
    module purge
fi
echo ""

echo "Run PCMCI for Rasp et al."
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