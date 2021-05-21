GLOO_SOCKET_IFNAME=ens5 \
TP_SOCKET_IFNAME=ens5 \
python driver.py \
    --use-gpu \
    --num-nodes 2 \
    --node-rank $1 \
    --num-trainers 4 \
    --master-addr q3-dy-p3dn24xlarge-4 \
    &> "out$1"
