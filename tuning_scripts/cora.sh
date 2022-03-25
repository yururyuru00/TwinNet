# we decide hyper-parameters based on page 7 of https://arxiv.org/pdf/1710.10903.pdf (transductive learning)

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m 'key=TwinGAT_Cora' \
     'TwinGAT_Cora.n_head=8' \
     'TwinGAT_Cora.n_head_last=1' \
     'TwinGAT_Cora.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_Cora.dropout=choice(0.,0.4,0.6)' \
     'TwinGAT_Cora.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_Cora.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'TwinGAT_Cora.n_hid=choice(8, 16)' \
     'TwinGAT_Cora.temparature=choice(0.1, 0.5, 1.0)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
