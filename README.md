For running single node experiments, run the following:
```
python driver.py
```

You can explore the options by passing the `--help` flag.
Use the `--use-gpu` flag to run the experiments on GPUs.
Certain flags have been set by default in the driver.py `main()` function for ease of experimentation.

For running multi-node experiments, a launcher script has been provided in `train.sh`.
To use it, you must edit the script and set the number of nodes you wish to use (`--num-nodes`), the address of the master node (`--master-addr`), the total number of trainer processes desired (`--num-trainers`), and the network interface TensorPipe should use (setting the env vars `GLOO_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` to the output of `echo $(ip r | grep default | awk '{print $5}')`).
Then you can simply run the following on each node:
```
./train.sh <node_rank>
```

where `<node_rank>` is a unique rank for each node (0 ... num\_nodes-1).
