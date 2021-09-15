#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=compute    # Specify partition name
#SBATCH --nodes=1
##SBATCH --threads-per-core=2
##SBATCH --ntasks=1             # Specify max. number of tasks to be invoked
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0
#SBATCH --exclusive            # node not shared
#SBATCH --job-name=Ra_Tr

#=============================================================================

# HISTORY:
# 09-Feb-2021 Individual parents/children (levels),      FIS
# 05-Feb-2021 prepost partition,                         FIS
# 29-Jul-2020 Written,                                   FIS
#
# DESCRIPTION:
#
# NOTES:
# Reserve node:
#    $> salloc --partition=compute --nodes=1 --time=2:00:00 --account $PROJECT
#    $> ssh <allocated node>
# Activate conda environment:
#    $> conda activate tg38plus
# 
#=============================================================================

# File to be processed:
# 2021_09_02_TRAIN_For_Nando.nc; 2021_09_02_TRAIN_For_Nando_shuffle.nc
# 2021_09_02_VALID_For_Nando.nc
# 2021_09_02_TEST_For_Nando.nc; 2021_09_02_TEST_For_Nando_shuffle.nc
casenm=TRAIN # TRAIN; VALID; TEST
filenm=2021_09_02_${casenm}_For_Nando.nc

# Variables
#spcam_parents      = ['tbp','qbp','vbp','ps','solin','shflx','lhflx']\n",
#spcam_children     = ['tphystnd','phq', 'prect','fsnt','fsns','flnt','flns']
#variables=' qbp tbp vbp ps solin shflx lhflx tphystnd phq prect fsnt fsns flnt flns qrl qrs '
variables=' qbp tbp vbp ps solin shflx lhflx phq tphystnd fsnt fsns flnt flns prect '

declare -A varsIDX
varsIDX[qbp]="0,29"; varsIDX[tbp]="30,59"; varsIDX[vbp]="60,89"; varsIDX[ps]="90,90"
varsIDX[solin]="91,91"; varsIDX[shflx]="92,92"; varsIDX[lhflx]="93,93"; varsIDX[phq]="94,123"
varsIDX[tphystnd]="124,153"; varsIDX[fsnt]="154,154"; varsIDX[fsns]="155,155" 
varsIDX[flnt]="156,156"; varsIDX[flns]="157,157"; varsIDX[prect]="158,158"


# Paths
scriptPath=`pwd`
tmpPath=${scriptPath}/tmp
dataPath=${scriptPath}/SPCAM_tb_preproc

# SCRIPTS
pyScript=spcam_tb_prepro_v2.py

# Environment
source /pf/b/b309172/.bashrc
conda activate causalnncam

# Loop through variables
echo "Reconstructing SPCAM variables for $filenm"
for iVar in $variables ; do
    
    echo "key  : $iVar; value: ${varsIDX[$iVar]}"

    # outPath?
    outPath=${scriptPath}/outPath_rasp/$iVar
    if [ ! -d $outPath ]; then
	mkdir -p $outPath
    fi
    ln -s $dataPath/$filenm $outPath/$filenm
    
    IFS=',' read -ra ADDR <<< "${varsIDX[$iVar]}"
    typeset -i i END count
    let END=${ADDR[1]} i=${ADDR[0]} count=1
#    let END=${ADDR[1]} i=${ADDR[1]} count=1 # For testing
    while ((i<=END)); do
	
	echo "$count $i"
	
	outFile=${iVar}_${count}_${filenm}
	if [ ! -f $outPath/$outFile ]; then
	    
   	    # Extract iVar
	    echo "... extract..."
	    ncks -d var_names,${i},${i} $outPath/$filenm $outPath/${iVar}_${count}_${casenm}.nc

            # Run python
	    echo "... reconstruct..."
	    python $pyScript ${iVar}_${count}_${casenm}.nc $outPath $outPath "$iVar" $count $casenm
	    mv $outPath/${iVar}_${iVar}_${count}_${casenm}.nc $outPath/$outFile
	    rm $outPath/${iVar}_${count}_${casenm}.nc
	    echo ""
	  
	else	
	    echo "$outFile exists; skipping..."
	    echo ""
  
	fi

	let i++ count++
	
    done

    # Clean-up
    rm $outPath/$filenm

done
conda deactivate

