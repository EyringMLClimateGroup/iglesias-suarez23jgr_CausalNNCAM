#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=compute    # Specify partition name
#SBATCH --nodes=1
##SBATCH --threads-per-core=1
##SBATCH --ntasks=1             # Specify max. number of tasks to be invoked
#SBATCH --time=00:20:00        # Set a limit on the total run time
#SBATCH --mem=0
#SBATCH --exclusive            # node not shared
#SBATCH --job-name=spcam_prepro_var

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

# File to be processed (002_train_1_month.nc; 003_train_2_month.nc; 002_train_1_year.nc)
filenm=002_train_1_month.nc
#filenm=003_train_2_month.nc
#filenm=002_train_1_year.nc

# Variables
#spcam_parents      = ['tbp','qbp','vbp','ps','solin','shflx','lhflx']\n",
#spcam_children     = ['tphystnd','phq', 'prect','fsnt','fsns','flnt','flns']
variables=' tbp qbp vbp ps solin shflx lhflx tphystnd phq prect fsnt fsns flnt flns qrl qrs '
#variables=' ps solin shflx lhflx prect fsnt fsns flnt flns '
#variables=' tbp qbp vbp tphystnd phq '
#variables=' qrl qrs '
variables=' ps '

declare -A varsIDX
varsIDX[qbp]="0,29"; varsIDX[qcbp]="30,59"; varsIDX[qibp]="60,89"; varsIDX[tbp]="90,119"
varsIDX[vbp]="120,149"; varsIDX[ps]="150,150"; varsIDX[solin]="151,151"; varsIDX[shflx]="152,152" 
varsIDX[lhflx]="153,153"; varsIDX[phq]="154,183"; varsIDX[phcldliq]="184,213"; 
varsIDX[phcldice]="214,243"; varsIDX[tphystnd]="244,273"; varsIDX[qrl]="274,303"; 
varsIDX[qrs]="304,333"; varsIDX[dtvke]="334,363"; varsIDX[fsnt]="364,364"; varsIDX[fsns]="365,365"
varsIDX[flnt]="366,366"; varsIDX[flns]="367,367"; varsIDX[prect]="368,368"; varsIDX[prectend]="369,369" 
varsIDX[precst]="370,370"; varsIDX[precsten]="371,371"; varsIDX[qdt_adiabatic]="372,401" 
varsIDX[qcdt_adiabatic]="402,431"; varsIDX[qidt_adiabatic]="432,461"; varsIDX[tdt_adiabatic]="462,491" 
varsIDX[vdt_adiabatic]="492,521"



# Paths
scriptPath=`pwd`
dataPath=/work/bd1179/b309172/data/SPCAM_gb_preproc

# SCRIPTS
pyScript=spcam_preproc_var_v2.py

# Environment
source /pf/b/b309172/.bashrc
conda activate causalnncam

# Loop through variables
echo "Reconstructing SPCAM variables for $filenm"
for iVar in $variables ; do
    
    echo "key  : $iVar; value: ${varsIDX[$iVar]}"

    # outPath?
    outPath=/work/bd1179/b309172/data/SPCAM_gb_recons/$iVar
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
	    ncks -d var_names,${i},${i} $outPath/$filenm $outPath/${iVar}_${count}.nc

            # Run python
	    echo "... reconstruct..."
	    python $pyScript ${iVar}_${count}.nc $outPath $outPath "$iVar" $count
	    mv $outPath/${iVar}_${iVar}_${count}.nc $outPath/$outFile
	    rm $outPath/${iVar}_${count}.nc
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