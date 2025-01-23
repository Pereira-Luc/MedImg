#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=monai_testing
#SBATCH --output=test_output.log
#SBATCH --error=test_error.log
#SBATCH --qos=normal

module load math/CPLEX/22.11-GCCcore-10.2.0-Python-3.8.6
module load lib/PyYAML/5.3.1-GCCcore-10.2.0

pip install --user montai nibabel itk sympy --quiet
pip install --user --upgrade numpy --quiet

python3 -c "import monai; print ('MONAI version:', monai.__version__)"

python3 test.py
