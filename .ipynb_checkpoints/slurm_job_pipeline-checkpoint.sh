#!/bin/bash
# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=pipeline
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --threads-per-core=2
#SBATCH --mem=0
#SBATCH --output=LOG.pipeline-%j.o
#SBATCH --error=LOG.pipeline-%j.o
##SBATCH --exclusive
#SBATCH --mail-type=FAIL
#SBATCH --time=00:30:00


# Paths
scriptPath=`pwd`

# Scripts
pyScript=pipeline
cfgFile=cfg_${pyScript}.yml

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

cd $scriptPath
echo "Convert jupyter notebook into python script"
module load python3/unstable
jupyter nbconvert --to script ${pyScript}.ipynb
module purge
echo ""

echo "Run PCMCI"
source /pf/b/b309172/.bashrc
conda activate tg38plus
python ${pyScript}.py -c $cfgFile

# Clean-up
rm ${pyScript}.py

echo ""
echo "---------- Finished $0 ----------"