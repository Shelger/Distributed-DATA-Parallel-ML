import os
import torch

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.barrier()

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not Find GPU device for training.")

    # initialize environment
    init_distributed_mode(args=args)
    rank = args.rank
    device = args.device(args.device)
    batch_size = args.batch_size
    num_classes = args.num_classes
    weights_path = args.weights
    args.lr *= args.world