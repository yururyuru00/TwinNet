IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_CiteSeer_tuning 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_CiteSeer.n_layer=choice(2,4,6)' \
     'TwinGCN_CiteSeer.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_CiteSeer.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinGCN_CiteSeer.weight_decay=choice(0,0.0001,0.0005,0.001)' \
     'TwinGCN_CiteSeer.scope=global' \
     'TwinGCN_CiteSeer.kernel=wdp' \
     'TwinGCN_CiteSeer.temparature=choice(-0.1,-0.5,-1)' \


     python3 train.py -m mlflow.runname=TwinGAT_PubMed_tuning 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_PubMed.n_layer=choice(2,4,6)' \
     'TwinGAT_PubMed.dropout=choice(0.,0.2,0.4)' \
     'TwinGAT_PubMed.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinGAT_PubMed.weight_decay=choice(0,0.0001,0.0005,0.001)' \
     'TwinGAT_PubMed.scope=local' \
     'TwinGAT_PubMed.kernel=wdp' \
     'TwinGAT_PubMed.temparature=choice(0.1,0.5,1)' \

     python3 train.py -m mlflow.runname=TwinGAT_PubMed_tuning 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_PubMed.n_layer=choice(2,4,6)' \
     'TwinGAT_PubMed.dropout=choice(0.,0.2,0.4)' \
     'TwinGAT_PubMed.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinGAT_PubMed.weight_decay=choice(0,0.0001,0.0005,0.001)' \
     'TwinGAT_PubMed.scope=global' \
     'TwinGAT_PubMed.kernel=wdp' \
     'TwinGAT_PubMed.temparature=choice(-0.1,-0.5,-1)' \
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done