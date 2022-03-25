# we decide hyper-parameters based on page 7 of https://arxiv.org/pdf/1710.10903.pdf (inductive learning)

IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.n_head=4' \
     'TwinGAT_PPIinduct.n_head_last=6' \
     'TwinGAT_PPIinduct.skip_connection=choice(vanilla,res,dense,highway)' \
     'TwinGAT_PPIinduct.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_PPIinduct.dropout=choice(0.,0.4,0.6)' \
     'TwinGAT_PPIinduct.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_PPIinduct.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'TwinGAT_PPIinduct.n_hid=choice(8, 16)' \
     'TwinGAT_PPIinduct.temparature=choice(0.1, 0.5, 1.0)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
