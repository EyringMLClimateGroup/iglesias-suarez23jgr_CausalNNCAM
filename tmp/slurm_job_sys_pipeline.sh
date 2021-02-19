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
cfgFile='cfg_pipeline.yml'
pyScript=sys_pipeline

# Options
#spcam_parents=['tbp','qbp','vbp','ps','solin','shflx','lhflx']
spcam_parents=['tbp','ps']
#spcam_children=['tphystnd','prect', 'fsns', 'flns']
spcam_children=['flns']
region=[['4','4'],['120','120']]
lim_levels=['850','700']
target_levels=None
pc_alphas=['0.001','0.002','0.005','0.01','0.02','0.05','0.1','0.2']
pc_alphas="0.001 0.002"

## PROCESSING
#
echo "---------- Starting $0 ----------"
echo ""

echo "Run PCMCI"
#python ${pyScript}.py -p $spcam_parents -c $spcam_children -a $pc_alphas -r $region -l $lim_levels
python ${pyScript}.py -c $cfgFile 

echo ""
echo "---------- Finished $0 ----------"