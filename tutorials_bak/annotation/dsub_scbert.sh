#!/bin/bash
#DSUB -n scbert_anno
#DSUB -N 1
#DSUB -l p20240507
#DSUB -A root.project.P24Z28400N0258
#DSUB -R "cpu=24;gpu=4;mem=64000"
#DSUB -oo ./logs/scbert_anno.out.%J
#DSUB -eo ./logs/scbert_anno.err.%J


HOST=`hostname`
date=`date +%Y-%m-%d-%H-%M-%S`
HOSTFILE="./logs/hostnames.log."$date
flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_IP=`head -n 1 ${HOSTFILE}`
HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=${HOST_RANK}-1
NNODES=1
NUM_GPU_PER_NODE=4
host_name=`hostname`
echo 'hostname:'${host_name}


##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

##Config nnodes node_rank master_addr
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11
module load libs/nccl/2.18.3_cuda11
module load compilers/gcc/12.3.0

cd /home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/tutorials/annotation/;
torchrun --nproc_per_node=4 --master_port=2222 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_IP \
 ./run_scbert.py ./configs/organs/scbert_lung.toml
