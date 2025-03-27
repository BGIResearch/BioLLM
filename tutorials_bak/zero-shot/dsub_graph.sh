#!/bin/bash
#DSUB -n construct_graph
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0258
#DSUB -R "cpu=24;gpu=1;mem=150000"
#DSUB -oo logs/construct_graph.out.%J
#DSUB -eo logs/construct_graph.err.%J

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11
module load libs/nccl/2.18.3_cuda11
module load compilers/gcc/12.3.0


JOB_PATH="/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/tutorials/zero-shot"
cd ${JOB_PATH}
python -u ./construct_graph.py
