IFS_BACKUP=$IFS
IFS=$'\n'
ary=("

     python3 train.py -m mlflow.runname=TwinGCN_Cornell 'key=TwinGCN_Cornell' \
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
     
     python3 train.py -m mlflow.runname=TwinGCN_Texas 'key=TwinGCN_Texas' \
     'TwinGCN_Texas.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_Texas.n_layer=range(2,9)' \
     'TwinGCN_Texas.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGCN_Texas.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGCN_Texas.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGCN_Texas.n_hid=choice(32,64)'
     'TwinGCN_Texas.temparature=interval(-1.,1.)' \
     'TwinGCN_Texas.scope=choice(local,global)' \
     'TwinGCN_Texas.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGCN_Texas.activation=choice(ReLU,Identity)' \
     'TwinGCN_Texas.self_loop=choice(True,False)'
     
     python3 train.py -m mlflow.runname=TwinGCN_Wisconsin 'key=TwinGCN_Wisconsin' \
     'TwinGCN_Wisconsin.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_Wisconsin.n_layer=range(2,9)' \
     'TwinGCN_Wisconsin.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGCN_Wisconsin.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGCN_Wisconsin.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGCN_Wisconsin.n_hid=choice(32,64)'
     'TwinGCN_Wisconsin.temparature=interval(-1.,1.)' \
     'TwinGCN_Wisconsin.scope=choice(local,global)' \
     'TwinGCN_Wisconsin.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGCN_Wisconsin.activation=choice(ReLU,Identity)' \
     'TwinGCN_Wisconsin.self_loop=choice(True,False)'
     
     
     python3 train.py -m mlflow.runname=TwinGAT_Cornell 'key=TwinGAT_Cornell' \
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
     
     python3 train.py -m mlflow.runname=TwinGAT_Texas 'key=TwinGAT_Texas' \
     'TwinGAT_Texas.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_Texas.n_layer=range(2,9)' \
     'TwinGAT_Texas.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_Texas.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_Texas.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGAT_Texas.n_hid=choice(32,64)'
     'TwinGAT_Texas.temparature=interval(-1.,1.)' \
     'TwinGAT_Texas.scope=choice(local,global)' \
     'TwinGAT_Texas.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGAT_Texas.activation=choice(ELU,Identity)' \
     'TwinGAT_Texas.self_loop=choice(True,False)'
     
     python3 train.py -m mlflow.runname=TwinGAT_Wisconsin 'key=TwinGAT_Wisconsin' \
     'TwinGAT_Wisconsin.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_Wisconsin.n_layer=range(2,9)' \
     'TwinGAT_Wisconsin.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_Wisconsin.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_Wisconsin.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGAT_Wisconsin.n_hid=choice(32,64)'
     'TwinGAT_Wisconsin.temparature=interval(-1.,1.)' \
     'TwinGAT_Wisconsin.scope=choice(local,global)' \
     'TwinGAT_Wisconsin.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGAT_Wisconsin.activation=choice(ELU,Identity)' \
     'TwinGAT_Wisconsin.self_loop=choice(True,False)'

    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
