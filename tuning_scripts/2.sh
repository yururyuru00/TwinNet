IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinRGCN_MAG_tuning 'key=TwinRGCN_MAG' \
     'TwinRGCN_MAG.n_layer=choice(2,3)' \
     'TwinRGCN_MAG.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinRGCN_MAG.dropout=choice(0.,0.2,0.5)' \
     'TwinRGCN_MAG.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinRGCN_MAG.weight_decay=choice(0,1E-4,1E-3)' \
     'TwinRGCN_MAG.scope=local' \
     'TwinRGCN_MAG.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinRGCN_MAG_tuning 'key=TwinRGCN_MAG' \
     'TwinRGCN_MAG.n_layer=choice(2,3)' \
     'TwinRGCN_MAG.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinRGCN_MAG.dropout=choice(0.,0.2,0.5)' \
     'TwinRGCN_MAG.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinRGCN_MAG.weight_decay=choice(0,1E-4,1E-3)' \
     'TwinRGCN_MAG.scope=global' \
     'TwinRGCN_MAG.temparature=choice(-0.1,-0.5,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done


  norm: None
  n_layer: 2
  n_hid: 64
  dropout: 0.5
  learning_rate: 0.01
  weight_decay: 0