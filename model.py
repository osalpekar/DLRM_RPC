import sys
import time
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.rpc import RRef, rpc_sync
from torch.nn.parallel import DistributedDataParallel as DDP

RANDOM_SEED = 100

def set_rand_seed():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def select_device(use_gpu, rank=0):
    if use_gpu and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        return torch.device("cuda", rank % num_devices)
    else:
        return torch.device("cpu")

def stamp_time(cuda=False):
    if cuda:
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(0))
        return event
    else:
        return time.time()

def compute_delay(ts, cuda=False):
    if cuda:
        return ts["tik"].elapsed_time(ts["tok"]) / 1e3
    else:
        return ts["tok"] - ts["tik"]

class MLP(nn.Module):
    """
    MLP module consisting of several linear layers following by activations.
    These are the dense parameters local to each trainer and will be synced via
    DDP.
    """
    def __init__(self, ln, sigmoid_layer, use_gpu, rank, name=None):
        super(MLP, self).__init__()

        set_rand_seed()
        self.device = select_device(use_gpu, rank)
        print("Using {} for {} ...".format(self.device, name))

        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)
            LL = LL.to(self.device)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid().to(self.device))
            else:
                layers.append(nn.ReLU().to(self.device))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        return self.mlp(x).cpu()


class Emb(nn.Module):
    """
    This is the embedding module, which is placed on the parameter server(s).
    The trainers will issue RPC's to this module for embedding lookups. This
    layer will get grad updates using Distributed Autograd and Distributed
    Optimizer.
    """
    def __init__(self, embedding_dim, num_embeddings, emb_tab_ids, use_gpu, rank):
        super(Emb, self).__init__()

        set_rand_seed()
        self.device = select_device(use_gpu, rank)
        print("Using {} for Embeddings ...".format(self.device))

        self.emb_tab_ids = emb_tab_ids

        emb_l = nn.ModuleList()
        for i in emb_tab_ids:
            n = num_embeddings[i]

            EE = nn.EmbeddingBag(n, embedding_dim, mode="sum", sparse=True)
            EE = EE.to(self.device)
            emb_l.append(EE)

        self.emb_l = emb_l

    def forward(self, offsets, indices):
        outputs = []

        for ind, val in enumerate(self.emb_tab_ids):
            sparse_indices = indices[val]
            sparse_offsets = offsets[val]

            Embedding = self.emb_l[ind]

            Vec = Embedding(
                sparse_indices.to(self.device),
                sparse_offsets.to(self.device)
            )

            outputs.append(Vec.cpu())

        return outputs, self.emb_tab_ids


class DLRM_RPC(nn.Module):
    """
    This is the overall model. Each replica holds 2 MLP modules and a list of
    RRefs to the embeddings on the Parameter Server.

    The training flow is as follows:
        1. Each replica gets a batch of data and issues RPCs to do embedding
           lookups from the PS embedding tables.
        2. Each replica performs the forward pass on the locally held bottom
           bottom MLP module.
        3. It interacts the outputs of the MLP module and the retreived
           embeddings to create a combined feature representation.
        4. The combined features from the previous step are passed through the
           locally help top MLP, which forms the final model output.

    The dense layers (top and bottom MLP) are synchronized via DDP. The sparse
    embeddings are updated via Distributed Autograd (using RPCs to
    communicate gradients).
    """
    def __init__(
        self,
        embedding_rref_list,
        rank,
        use_gpu,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
    ):
        super(DLRM_RPC, self).__init__()
        set_rand_seed()

        self.device = select_device(use_gpu, rank)
        self.use_gpu = use_gpu

        self.arch_interaction_op = arch_interaction_op
        self.arch_interaction_itself = arch_interaction_itself
        self.loss_threshold = 0.0

        self.emb_rref = embedding_rref_list
        self.ln = ln_emb

        top_mlp = MLP(ln_top, sigmoid_top, use_gpu, rank, name="top_mlp")
        bot_mlp = MLP(ln_bot, sigmoid_bot, use_gpu, rank, name="bot_mlp")

        device_ids = [rank] if use_gpu else None
        self.top_mlp_ddp = DDP(top_mlp.to(self.device), device_ids=device_ids)
        self.bot_mlp_ddp = DDP(bot_mlp.to(self.device), device_ids=device_ids)

    def forward(self, dense_x, offsets, indices):
        """
        dense_x is the dense input that is passed through the bottom MLP.
        offsets and indices are used for Embedding Lookups (via RPC).
        """

        timestamps = {}
        def mark_complete_cpu(index, cuda, fut):
            timestamps[index]["tok"] = stamp_time(cuda)

        start = time.time()
        embedding_output = [0 for i in range(0, self.ln.size)]
        futs = []
        index = 0
        for emb_rref in self.emb_rref:
            # TODO: can we make this part async so it overlaps with the bottom
            # MLP forward? I'm thinking we can make this an rpc_async that
            # returns a future. We can attach a callback to each future that
            # populates the embedding_output list. Then we wait for all futures
            # after the MLP forward call.
            timestamps[index] = {}
            timestamps[index]["tik"] = stamp_time(self.use_gpu)
            fut = emb_rref.rpc_async().forward(
                offsets,
                indices,
            )
            fut.add_done_callback(partial(mark_complete_cpu, index, self.use_gpu))
            futs.append(fut)
            index += 1

        torch.futures.wait_all(futs)
        end = time.time()

        # NOTE: The CUDA Synchronize makes this code really slow but allows us to
        # measure RPC and embedding lookup. When benchmarking the whole thing,
        # maybe we can turn this if check to False (will not measure comp/comms
        # but total throughput is more accurate).
        if False:
            if self.use_gpu:
                 torch.cuda.synchronize(0)

            delays = []
            for index in range(len(timestamps)):
                delays.append(compute_delay(timestamps[index], self.use_gpu))
        else:
            delays = [0]

        for fut in futs:
            embedding_lookup, inds = fut.value()
            #TODO: you can make this part a callback
            for ind, val in enumerate(inds):
                embedding_output[val] = embedding_lookup[ind]

        # pass input to the remote bottom MLP and get the output back
        x = self.bot_mlp_ddp(dense_x)
        # interact features (dense and sparse)
        z = self.interact_features(x, embedding_output)

        # pass input to the remote top MLP and get the output back
        p = self.top_mlp_ddp(z)
        # clamp to [0,1] to represent probability of click
        z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))

        return z, delays, end-start

    def interact_features(self, x, embedding_output):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + embedding_output, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + embedding_output, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R
