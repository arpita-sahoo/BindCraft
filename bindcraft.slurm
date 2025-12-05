#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 48gb
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM32GB|GPUMEM80GB
#SBATCH --time 12:00:00
#SBATCH --output=bindcraft_%A.log
##SBATCH --mail-user=user@example.com
##SBATCH --mail-type=BEGIN,END 

# Initialise environment and modules
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /shares/seeger.imm.uzh/BindCraft-env
CONDA_BASE=$(conda info --base)

# Parsing command line options
SETTINGS=""
FILTERS=""
ADVANCED=""
TEMP=$(getopt -o s:f:a: --long settings:,filters:,advanced: -n 'bindcraft.slurm' -- "$@")
eval set -- "$TEMP"

while true ; do
    case "$1" in
        -s|--settings) SETTINGS="$2" ; shift 2 ;;
        -f|--filters) FILTERS="$2" ; shift 2 ;;
        -a|--advanced) ADVANCED="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Invalid Option" ; exit 1 ;;
    esac
done

# Ensure that SETTINGS is not empty
if [ -z "$SETTINGS" ]; then
    echo "Error: The -s or --settings option is required."
    exit 1
fi

echo "Running the BindCraft pipeline"
python -u ./bindcraft.py --settings "${SETTINGS}" --filters "${FILTERS}" --advanced "${ADVANCED}"
