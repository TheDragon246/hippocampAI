#!/bin/sh
# The job name: you can choose whatever you want for this.
#SBATCH --job-name=test_job
#SBATCH --account=stevenweisberg
#SBATCH --qos=stevenweisberg

# Your email address and the events for which you want to receive email
# notification (NONE, BEGIN, END, FAIL, ALL).
#SBATCH --mail-user=ashishkumarsahoo@ufl.edu
#SBATCH --mail-type=NONE

# The compute configuration for the job. For a job that uses GPUs, the
# partition must be set to "gpu". This example script requests access
# to a single GPU, 16 CPUs, and 30 GB of RAM for a single PyTorch task.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000mb

# Specifies how long the job will be allowed to run in HH:MM:SS.
#SBATCH --time=144:00:00

# The log file for all job output. Note the special string "%j", which
# represents the job number.
#SBATCH --output=job_outputs/job_output_standardized_normalized_hippocampus_intensity_brain_181023_%j.out

# Prints the working directory, name of the assigned node, and
# date/time at the top of the output log.
#pwd; hostname; date


#Mask the ROIs and make sure all the data is saved in the proper location
module load nilearn
#python mask.py

echo 'standardized_normalized_hippocampus_intensity_brain_181023.py'
CUDA_LAUNCH_BLOCKING=1 python standardized_normalized_hippocampus_intensity_brain_181023.py
