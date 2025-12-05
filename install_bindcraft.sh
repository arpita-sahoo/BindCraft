#!/bin/bash
#########################################
# BindCraft Full Environment Rebuild
#########################################

set -e  # Exit immediately on error

SECONDS=0

#########################################
# Load Conda
#########################################
module load miniforge3
source $(conda info --base)/etc/profile.d/conda.sh

ENV_PATH="/home/$USER/bindcraft-env"

echo ">>> Creating / activating conda environment at: $ENV_PATH"
if [ ! -d "$ENV_PATH" ]; then
    conda create -y -p "$ENV_PATH" python=3.10
fi
conda activate "$ENV_PATH"

#########################################
# Install required Python dependencies
# NOTE: This MUST be run on a GPU compute node !
#########################################
echo ">>> Installing Python dependencies (GPU-compatible JAX)"

# Update pip
pip install --upgrade pip

pip install --no-cache-dir \
    numpy pandas scipy matplotlib seaborn \
    biopython tqdm joblib \
    chex dm-haiku dm-tree optax immutabledict ml-collections \
    py3dmol fsspec

#########################################
# Install GPU-JAX
#########################################
echo ">>> Installing CUDA-enabled JAX"
pip install --no-cache-dir "jax[cuda]" jaxlib

#########################################
# Install ColabDesign from GitHub
#########################################
echo ">>> Installing ColabDesign"
pip install --no-cache-dir --no-deps git+https://github.com/sokrypton/ColabDesign.git

python - << EOF
import jax, colabdesign
print("✔ Python packages OK")
print("Detected devices:", jax.devices())
EOF

#########################################
# Download AlphaFold2 weights
#########################################
echo ">>> Downloading AlphaFold2 weights"
INSTALL_DIR=$(pwd)
PARAMS_DIR="${INSTALL_DIR}/params"
PARAMS_FILE="${PARAMS_DIR}/alphafold_params_2022-12-06.tar"

mkdir -p "${PARAMS_DIR}"

if [ ! -f "${PARAMS_DIR}/params_model_5_ptm.npz" ]; then
    wget -O "${PARAMS_FILE}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    tar -xvf "${PARAMS_FILE}" -C "${PARAMS_DIR}"
    rm "${PARAMS_FILE}"
else
    echo ">>> Weights already extracted — skipping download."
fi

#########################################
# Fix executable permissions
#########################################
echo ">>> Setting permissions for DSSP + DAlphaBall"
chmod +x "${INSTALL_DIR}/functions/dssp" || echo "Warning: dssp not found"
chmod +x "${INSTALL_DIR}/functions/DAlphaBall.gcc" || echo "Warning: DAlphaBall.gcc not found"

#########################################
# Done
#########################################
conda deactivate

t=$SECONDS
echo -e "\n✔ BindCraft installation COMPLETE"
echo -e "Time: $(($t / 3600))h $((($t / 60) % 60))m $(($t % 60))s"
echo -e "\nActivate your environment using:"
echo -e "  module load miniforge3"
echo -e "  source \$(conda info --base)/etc/profile.d/conda.sh"
echo -e "  conda activate $ENV_PATH\n"
