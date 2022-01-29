IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     # remain 900 tri tunint with optuna
     python3 train.py -m mlflow.runname=TwinGAT_CiteSeer 'key=TwinGAT_CiteSeer' \
     'TwinGAT_CiteSeer.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_CiteSeer.n_layer=range(2,9)' \
     'TwinGAT_CiteSeer.dropout_att=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_CiteSeer.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_CiteSeer.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_CiteSeer.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGAT_CiteSeer.n_hid=choice(16,32)' \
     'TwinGAT_CiteSeer.temparature=interval(-1.,1.)' \
     'TwinGAT_CiteSeer.scope=choice(local,global)' \
     'TwinGAT_CiteSeer.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGAT_CiteSeer.activation=choice(ReLU,Identity)' \
     'TwinGAT_CiteSeer.self_loop=choice(True,False)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
