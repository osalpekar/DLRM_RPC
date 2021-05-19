GLOO_SOCKET_IFNAME=ens3 \
TP_SOCKET_IFNAME=ens3 \
python driver.py \
    --use-gpu \
    --num-nodes 2 \
    --node-rank $1 \
    --num-trainers 4 \
    --master-addr q2-dy-p38xlarge-1 \
    &> "out$1"
