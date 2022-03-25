# we decide hyper-parameters based on page 15 of https://arxiv.org/pdf/2006.07739.pdf

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m 'key=TwinSAGE_Arxiv' \
     'TwinSAGE_Arxiv.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinSAGE_Arxiv.dropout=choice(0.1,0.4,0.6)' \
     'TwinSAGE_Arxiv.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinSAGE_Arxiv.weight_decay=0' \
     'TwinSAGE_Arxiv.n_hid=choice(64, 128, 256)' \
     'TwinSAGE_Arxiv.temparature=choice(0.1, 0.5, 1.0)' \
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
