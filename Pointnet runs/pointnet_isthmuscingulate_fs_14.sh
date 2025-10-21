#!/bin/sh
# The job name: you can choose whatever you want for this.
#SBATCH --job-name=test_job
#SBATCH --account=stevenweisberg
#SBATCH --qos=stevenweisberg

# Your email address and the events for which you want to receive email
# notification (NONE, BEGIN, END, FAIL, ALL).
#SBATCH --mail-user=ashishkumarsahoo@ufl.edu
#SBATCH --mail-type=END

# The compute configuration for the job. For a job that uses GPUs, the
# partition must be set to "gpu". This example script requests access
# to a single GPU, 16 CPUs, and 30 GB of RAM for a single PyTorch task.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000mb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1


# Specifies how long the job will be allowed to run in HH:MM:SS.
#SBATCH --time=12:00:00

# The log file for all job output. Note the special string "%j", which
# represents the job number.
#SBATCH --output=job_outputs/job_output_isthmuscingulate_fs_14_%j.out

# Prints the working directory, name of the assigned node, and
# date/time at the top of the output log.
#pwd; hostname; date


#Mask the ROIs and make sure all the data is saved in the proper location
#module load nilearn
#python mask.py

#Mask the ROIs and make sure all the data is saved in the proper location
#module load pytorch
export PATH=/blue/stevenweisberg/ashishkumarsahoo/hippocampAI/conda_pyg_070923
module load conda
module load gcc
conda activate /blue/stevenweisberg/ashishkumarsahoo/hippocampAI/conda_pyg_070923
#module load pytorch/1.7.1

# Run the Python script.
echo 'pointnet_isthmuscingulate_fs_14.py'
CUDA_LAUNCH_BLOCKING=1 python pointnet_isthmuscingulate_fs_14.py
