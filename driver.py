from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import time
import json
#import submitit

import numpy as np
import os
import socket

# data generation
sys.path.append(os.getcwd())
import data as dp

import torch

import argparse
import subprocess
import torch.multiprocessing as mp

from torch.distributed import rpc
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch import optim
import model
from torch.distributed.rpc import RRef, rpc_sync

RANDOM_SEED = 100

def set_rand_seed():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _fetch(x):
    return x.cpu()


def set_print_options(print_precision):
    np.set_printoptions(precision=print_precision)
    torch.set_printoptions(precision=print_precision)


def init_gpu(use_gpu):
    if use_gpu:
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

	
def _retrieve_embedding_parameters(emb_rref):
    return [RRef(p) for p in emb_rref.local_value().parameters()]


def distributed_args_setting(args):

    node_list = os.environ.get('SLURM_STEP_NODELIST')
    if node_list is None:
        node_list = os.environ.get('SLURM_JOB_NODELIST')
    if node_list is not None:
        hostnames = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', node_list]
        )
        args.master_addr = hostnames.split()[0].decode('utf-8')
        args.init_method_rpc = 'tcp://{host}:{port}'.format(
            host=args.master_addr,
            port=args.distributed_port,
        )
        ddp_port = int(args.distributed_port) + 1
        args.init_method_ddp = 'tcp://{host}:{port}'.format(
            host=args.master_addr,
            port=str(ddp_port),
        )
        nnodes = int(os.environ.get('SLURM_NNODES'))
        ntasks_per_node = os.environ.get('SLURM_NTASKS_PER_NODE')
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
        else:
            ntasks = int(os.environ.get('SLURM_NTASKS'))
            nnodes = int(os.environ.get('SLURM_NNODES'))
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)
        if ntasks_per_node == 1:
            assert args.world_size % nnodes == 0
            gpus_per_node = args.world_size // nnodes
            node_id = int(os.environ.get('SLURM_NODEID'))
            args.distributed_rank = node_id * gpus_per_node
        else:
            assert ntasks_per_node == args.world_size // nnodes
            args.distributed_no_spawn = True
            args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
            args.device_id = int(os.environ.get('SLURM_LOCALID'))

    return args


def arg_parser():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="""Train Parameter-Server RPC based Deep Learning
        Recommendation Model (DLRM)""")
    parser.add_argument("--num-trainers", type=int, default=4)
    parser.add_argument("--num-ps", type=int, default=1)
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )
    # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=True)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    # RPC
    parser.add_argument("--distributed-rank", type=int, default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument("--master-addr", type=str, default="",
        help="""Address of master. Master must be able to accept network
        traffic on the address + port. Leave it empty if you want to run
        on Slurm.""")
    parser.add_argument("--distributed-port", type=str, default="29500",
        help="""Port that master is listening on. Master must be able
        to accept network traffic on the host and port.""")


    args = parser.parse_args()

    args.world_size = args.num_trainers + args.num_ps + 1
    if args.master_addr == "":
        args = distributed_args_setting(args)

    #assert args.distributed_rank is not None, "must provide rank argument."

    if args.master_addr is not "":
        # This means SLURM is being used and all init args are set correctly
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ["MASTER_PORT"] = args.distributed_port
    else:
        # This means SLURM is NOT being used.
        # NOTE: This means you are restricted to single-node training.
        hostname = socket.gethostname()
        LOCAL_IP = socket.gethostbyname(hostname)
        RPC_PORT = "29500"
        DDP_PORT = "29501"
        os.environ['MASTER_ADDR'] = LOCAL_IP
        os.environ["MASTER_PORT"] = RPC_PORT
        args.init_method_rpc = 'tcp://{host}:{port}'.format(
            host=LOCAL_IP,
            port=RPC_PORT,
        )
        args.init_method_ddp = 'tcp://{host}:{port}'.format(
            host=LOCAL_IP,
            port=DDP_PORT,
        )

    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    return args


def run_trainer(args, emb_rref_list):
    """
    Trainer function to be run from each machine. This function:
        1. Performs some basic initialization steps.
        2. Prepares random data for training.
        3. Sanity checks cmd-line args such as embedding sizes and MLP layers
        4. Sets up the model, loss, and Distributed Optimizer
        5. Runs the Training Loop
    """

    ######## BASIC INITIALIZATION ########
    set_rand_seed()
    set_print_options(args.print_precision)

    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    init_gpu(args.use_gpu)
    #print(args)

    ######## PREPARE TRAINING DATA ########
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input and target at random
    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    m_den = ln_bot[0]
    train_data, train_loader = dp.make_random_data_and_loader(args, ln_emb, m_den)
    nbatches = args.num_batches if args.num_batches > 0 else len(train_loader)

    ######## PARSE CMD LINE ARGS ########
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    ######## SANITY CHECKS ########
    # Ensure feature sizes and MLP dimensions match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if m_spa != m_den_out:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(m_spa)
            + " does not match last dim of bottom mlp "
            + str(m_den_out)
        )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, (X, offsets, indices, T) in enumerate(train_loader):
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(indices[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(offsets)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in indices])
            print(T.detach().cpu().numpy())


    ######## TRAINING SETUP ########

    # Initialize the model (note we are passing the list of RRefs that point to
    # the remote embeddings).
    dlrm = model.DLRM_RPC(
        emb_rref_list,
        args.distributed_rank,
        args.use_gpu,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
    )

    # Specify the loss function
    loss_fn = torch.nn.MSELoss(reduction="mean")

    model_parameter_rrefs = []
    # RRefs for embeddings from PS
    for ind, emb_rref in enumerate(emb_rref_list):
        ps_name = "ps{}".format(ind)
        model_parameter_rrefs.extend(
            rpc.rpc_sync(ps_name, _retrieve_embedding_parameters, args=(emb_rref,))
        )
    # RRefs local to the model (MLP)
    for param in dlrm.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Build DistributedOptimizer.
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=args.learning_rate,
    )

    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0

    # Lists to track forward and backwad times per iteration
    fwd_times = []
    bwd_times = []

    ######## RUN TRAINING LOOP ########
    with torch.autograd.profiler.profile(enabled=args.enable_profiling, use_cuda=args.use_gpu) as prof:
        for epoch in range(args.nepochs):

            accum_time_begin = time_wrap(args.use_gpu)

            if args.mlperf_logging:
                previous_iteration_time = None

            for j, (X, offsets, indices, T) in enumerate(train_loader):

                if args.mlperf_logging:
                    current_time = time_wrap(args.use_gpu)
                    if previous_iteration_time:
                        iteration_time = current_time - previous_iteration_time
                    else:
                        iteration_time = 0
                    previous_iteration_time = current_time
                else:
                    t1 = time_wrap(args.use_gpu)

                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                # create distributed autograd context
                with dist_autograd.context() as context_id:
                    # Run forward pass
                    fwd_start = time_wrap(args.use_gpu)
                    Z = dlrm.forward(X, offsets, indices)
                    fwd_end = time_wrap(args.use_gpu)

                    # Compute Loss
                    E = loss_fn(Z, T)

                    # Run distributed backward pass
                    bwd_start = time_wrap(args.use_gpu)
                    dist_autograd.backward(context_id, [E])
                    bwd_end = time_wrap(args.use_gpu)

                    # Run distributed optimizer
                    opt.step(context_id)

                    if epoch >= args.warmup_epochs:
                        fwd_times.append(fwd_end - fwd_start)
                        bwd_times.append(bwd_end - bwd_start)

                # compute loss and accuracy
                L = E.detach().cpu().numpy()  # numpy array
                S = Z.detach().cpu().numpy()  # numpy array
                T = T.detach().cpu().numpy()  # numpy array
                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                if args.mlperf_logging:
                    total_time += iteration_time
                else:
                    t2 = time_wrap(args.use_gpu)
                    total_time += t2 - t1
                total_accu += A
                total_loss += L * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                should_test = (
                    (args.test_freq > 0)
                    and (args.data_generation == "dataset")
                    and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                # print time, loss and accuracy
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    gA = total_accu / total_samp
                    total_accu = 0

                    gL = total_loss / total_samp
                    total_loss = 0

                    str_run_type = "inference" if args.inference_only else "training"
                    print(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                            str_run_type, j + 1, nbatches, epoch, gT
                        )
                        + "loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
                    )

                    log_iter = nbatches * epoch + j + 1
                    # Uncomment the line below to print out the total time with overhead
                    # print("Accumulated time so far: {}" \
                    # .format(time_wrap(args.use_gpu) - accum_time_begin))
                    total_iter = 0
                    total_samp = 0

        # END TRAIN LOOP
        mean_fwd = 1000.0 * np.mean(fwd_times)
        mean_bwd = 1000.0 * np.mean(bwd_times)
        std_fwd = 1000.0 * np.std(fwd_times)
        std_bwd = 1000.0 * np.std(bwd_times)

        print("[Trainer {}] Average FWD Time (ms): {}".format(args.distributed_rank, mean_fwd))
        print("[Trainer {}] STD DEV FWD Time (ms): {}".format(args.distributed_rank, std_fwd))
        print("[Trainer {}] Average BWD Time (ms): {}".format(args.distributed_rank, mean_bwd))
        print("[Trainer {}] STD DEV BWD Time (ms): {}".format(args.distributed_rank, std_bwd))

    # profiling
    if args.enable_profiling:
        with open("dlrm_s_pytorch.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("./dlrm_s_pytorch.json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

def run(rank, world_size, args):
    """
    General purpose run function that inits RPC, runs training and shuts down
    RPC.

    Training Setup:
    If we use 4 Trainers:
    Rank 0-3: Trainers
    Rank 4: PS
    Rank 5: Master

    If we use 16 Trainers:
    Rank 0-15: Trainers
    Rank 16: PS
    Rank 17: Master
    """
    if args.distributed_rank is None:
        args.distributed_rank = rank

    numCudaDevices = torch.cuda.device_count()
    localRank = rank % numCudaDevices
    torch.cuda.set_device(localRank)

    #print("Rank {} is using GPU {}".format(rank, torch.cuda.current_device()))

    # Using different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.rpc_timeout = 100
    rpc_backend_options.init_method = args.init_method_rpc
    # TODO: This will need some changes in the Sharded PS case

    # Rank 5: MASTER
    if rank == (args.num_trainers + args.num_ps):

        # Init RPC
        rpc.init_rpc(
            "master",
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Build the Embedding tables on the Parameter Servers.
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_spa = args.arch_sparse_feature_size
        embedding_ids = set(range(0, ln_emb.size))
        # When we have a single PS, the PS rank is the # of trainers, since the
        # trainers have rank 0 through #trainers - 1.
        param_server_rank = args.num_trainers

        emb_rref_list = []
        ps_name = "ps{}".format(0)
        emb_rref = rpc.remote(
            ps_name,
            model.Emb,
            args=(m_spa, ln_emb, embedding_ids, args.use_gpu, param_server_rank),
        )
        emb_rref_list.append(emb_rref)
        #print("Embedding RRef created:")
        #print(emb_rref)

        # Run the training loop on the trainers.
        futs = []
        for trainer_rank in range(args.num_trainers):
            trainer_name = "trainer{}".format(trainer_rank)
            args.distributed_rank = trainer_rank
            fut = rpc.rpc_async(
                trainer_name, run_trainer, args=(args, emb_rref_list)
            )
            futs.append(fut)

        torch.futures.wait_all(futs)

    # Rank 0-3: Trainers
    elif rank >= 0 and rank < args.num_trainers:
        
        #backend = dist.Backend.NCCL if args.use_gpu else dist.Backend.GLOO
        backend=dist.Backend.GLOO
        # Init PG for DDP
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=args.num_trainers,
            init_method=args.init_method_ddp,
        )

        # Initialize RPC. Trainer just waits for RPCs from master.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    # Rank 4: Parameter Server
    elif rank >= args.num_trainers and rank < (args.num_trainers + args.num_ps):

        # Init RPC
        ps_name = "ps{}".format(rank - args.num_trainers)
        rpc.init_rpc(
            ps_name,
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server does nothing
        pass

    rpc.shutdown()


def main():
    args = arg_parser()
    args.num_trainers = 4
    args.num_ps = 1
    args.world_size = args.num_trainers + args.num_ps + 1
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    #torch.backends.cudnn.enabled = False

    mp.spawn(run, args=(args.world_size, args,), nprocs=args.world_size, join=True)

#def submit():
#    executor = submitit.AutoExecutor(folder="log")
#    executor.update_parameters(
#        timeout_min=10,
#        slurm_partition="learnfair",
#        nodes=1
#    )
#    job = executor.submit(main)
#    output = job.result()

if __name__ == "__main__":
    # Use main() to launch locally and submit() to launch on fair cluster
    main()
