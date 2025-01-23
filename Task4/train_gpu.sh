#!/bin/bash -l
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node 1
#SBATCH --export=All
#SBATCH --time=04:30:00           
#SBATCH --job-name=monai_training_gpu # Job name
#SBATCH --output=train_gpu_output.log # Log file for output
#SBATCH --error=train_gpu_error.log   # Log file for errors
#SBATCH --qos=normal              # Set quality of service to normal


# Load necessary modules
module load math/CPLEX/22.11-GCCcore-10.2.0-Python-3.8.6
module load lib/PyYAML/5.3.1-GCCcore-10.2.0

# Install Python dependencies if not already installed
pip install --user monai nibabel itk sympy --quiet
pip install --user --upgrade numpy --quiet

# Confirm MONAI installation and version
python3 -c "import monai; print('MONAI version:', monai.__version__)"

# Run the training script
python3 train.py


