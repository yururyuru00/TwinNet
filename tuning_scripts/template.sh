IFS_BACKUP=$IFS
IFS=$'\n'
ary=("

     python3 train.py -m mlflow.runname=TwinGCN_Cornell 'key=TwinGCN_Cornell' \
     'hydra.sweeper.n_trials=100' \
     'TwinGCN_Cornell.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_Cornell.n_layer=range(2,9)' \
     'TwinGCN_Cornell.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGCN_Cornell.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGCN_Cornell.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGCN_Cornell.n_hid=choice(32,64)'
     'TwinGCN_Cornell.temparature=interval(-1.,1.)' \
     'TwinGCN_Cornell.scope=choice(local,global)' \
     'TwinGCN_Cornell.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGCN_Cornell.activation=choice(ReLU,Identity)' \
     'TwinGCN_Cornell.self_loop=choice(True,False)'
     
     
     python3 train.py -m mlflow.runname=TwinGAT_Cornell 'key=TwinGAT_Cornell' \
     'hydra.sweeper.n_trials=100' \
     'TwinGAT_Cornell.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_Cornell.n_layer=range(2,9)' \
     'TwinGAT_Cornell.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_Cornell.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_Cornell.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGAT_Cornell.n_hid=choice(32,64)'
     'TwinGAT_Cornell.temparature=interval(-1.,1.)' \
     'TwinGAT_Cornell.scope=choice(local,global)' \
     'TwinGAT_Cornell.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGAT_Cornell.activation=choice(ELU,Identity)' \
     'TwinGAT_Cornell.self_loop=choice(True,False)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
