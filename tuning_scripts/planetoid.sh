IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_Cora_valid 'key=TwinGCN_Cora' \
     'hydra.sweeper.n_trials=300' \
     'TwinGCN_Cora.n_layer=range(2,6)' \
     'TwinGCN_Cora.scope=choice(local,global)' \
     'TwinGCN_Cora.kernel=choice(wdp,ad)' \
     'TwinGCN_Cora.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinGCN_Cora.activation=choice(ReLU,Identity)' \
     'TwinGCN_Cora.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinGCN_CiteSeer_valid 'key=TwinGCN_CiteSeer' \
     'hydra.sweeper.n_trials=800' \
     'TwinGCN_CiteSeer.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_CiteSeer.n_layer=range(2,9)' \
     'TwinGCN_CiteSeer.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGCN_CiteSeer.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGCN_CiteSeer.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGCN_CiteSeer.n_hid=choice(16,32)' \
     'TwinGCN_CiteSeer.scope=choice(local,global)' \
     'TwinGCN_CiteSeer.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGCN_CiteSeer.temparature=interval(-1.,1.)' \
     'TwinGCN_CiteSeer.activation=choice(ReLU,Identity)' \
     'TwinGCN_CiteSeer.self_loop=choice(True,False)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
