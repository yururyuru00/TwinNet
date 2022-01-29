IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGAT_PubMed_valid 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGAT_PubMed.n_layer=range(2,9)' \
     'TwinGAT_PubMed.dropout_att=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_PubMed.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGAT_PubMed.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGAT_PubMed.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGAT_PubMed.n_hid=choice(8,16)' \
     'TwinGAT_PubMed.temparature=interval(-1.,1.)' \
     'TwinGAT_PubMed.scope=choice(local,global)' \
     'TwinGAT_PubMed.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGAT_PubMed.activation=choice(ELU,Identity)' \
     'TwinGAT_PubMed.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinGCN_Cora_valid 'key=TwinGCN_Cora' \
     'TwinGCN_Cora.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_Cora.n_layer=range(2,9)' \
     'TwinGCN_Cora.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGCN_Cora.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGCN_Cora.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGCN_Cora.n_hid=choice(16,32)' \
     'TwinGCN_Cora.temparature=interval(-1.,1.)' \
     'TwinGCN_Cora.scope=choice(local,global)' \
     'TwinGCN_Cora.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGCN_Cora.activation=choice(ReLU,Identity)' \
     'TwinGCN_Cora.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinGCN_CiteSeer_valid 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'TwinGCN_CiteSeer.n_layer=range(2,9)' \
     'TwinGCN_CiteSeer.dropout=choice(0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)' \
     'TwinGCN_CiteSeer.learning_rate=choice(0.05,0.01,0.005,0.001)' \
     'TwinGCN_CiteSeer.weight_decay=choice(0,5E-6,1E-5,5E-5,1E-4,5E-4,1E-3,5E-3,1E-2)' \
     'TwinGCN_CiteSeer.n_hid=choice(16,32)' \
     'TwinGCN_CiteSeer.temparature=interval(-1.,1.)' \
     'TwinGCN_CiteSeer.scope=choice(local,global)' \
     'TwinGCN_CiteSeer.kernel=choice(dp,sdp,wdp,ad,mx)' \
     'TwinGCN_CiteSeer.activation=choice(ReLU,Identity)' \
     'TwinGCN_CiteSeer.self_loop=choice(True,False)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
