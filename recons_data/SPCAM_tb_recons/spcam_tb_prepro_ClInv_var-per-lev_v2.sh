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
#SBATCH --job-name=CI_Tr

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
# 2021_09_03_TRAIN_For_Nando_ClInv.nc; 2021_09_03_TRAIN_For_Nando_ClInv_shuffle.nc
# 2021_09_03_VALID_For_Nando_ClInv.nc; 2021_09_03_VALID_For_Nando_ClInv_shuffle.nc
# 2021_09_03_TEST_For_Nando_ClInv.nc; 2021_09_03_TEST_For_Nando_ClInv_shuffle.nc
casenm=TRAIN # TRAIN; TEST; VALID
filenm=2021_09_03_${casenm}_For_Nando_ClInv.nc

# Variables
#spcam_parents      = 
#spcam_children     = 
variables=' rh bmse ps solin shflx lhf_nsdelq phq tphystnd fsnt fsns flnt flns prect '
#variables=' rh '

declare -A varsIDX
varsIDX[rh]="0,29"; varsIDX[bmse]="30,59"; varsIDX[ps]="60,60"; varsIDX[solin]="61,61"
varsIDX[shflx]="62,62"; varsIDX[lhf_nsdelq]="63,63"; varsIDX[phq]="64,93"; varsIDX[tphystnd]="94,123"
varsIDX[fsnt]="124,124"; varsIDX[fsns]="125,125"; varsIDX[flnt]="126,126"; varsIDX[flns]="127,127"
varsIDX[prect]="128,128"


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
    outPath=/work/bd1179/b309172/data/SPCAM_tb_preproc/$iVar
    if [ ! -d $outPath ]; then
	mkdir -p $outPath
    fi
    ln -s $dataPath/$filenm $outPath/$filenm
    
    IFS=',' read -ra ADDR <<< "${varsIDX[$iVar]}"
    typeset -i i END count
    let END=${ADDR[1]} i=${ADDR[0]} count=1
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

