IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=range(2,5)' \
     'TwinSAGE_Reddit.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinSAGE_Reddit.dropout=choice(0.,0.2,0.5)' \
     'TwinSAGE_Reddit.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinSAGE_Reddit.weight_decay=choice(0,1E-4,1E-3)' \
     'TwinSAGE_Reddit.scope=local' \
     'TwinSAGE_Reddit.temparature=choice(0.1,0.5,1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done