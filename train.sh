GLOO_SOCKET_IFNAME=front0 \
TP_SOCKET_IFNAME=front0 \
python driver.py \
    --use-gpu \
    --num-nodes 3 \
    --node-rank $1 \
    --num-trainers 16 \
    --master-addr learnfair1216 \
    &> "out$1"
