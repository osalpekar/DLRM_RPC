GLOO_SOCKET_IFNAME=ens5 \
TP_SOCKET_IFNAME=ens5 \
python driver.py \
    --use-gpu \
    --num-nodes 3 \
    --node-rank $1 \
    --num-trainers 16 \
    --master-addr q3-dy-p3dn24xlarge-2 \
    &> "out$1"
