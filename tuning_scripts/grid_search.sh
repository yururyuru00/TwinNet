IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=GCN_Cora_L2-9 'key=TwinGCN_Cora' \
     'TwinGCN_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_Cora_L2-9 'key=GCN_Cora' \
     'GCN_Cora.skip_connection=choice(vanilla,res,dense,highway)' \
     'GCN_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_Cora_L2-9 'key=JKGCN_Cora' \
     'JKGCN_Cora.jk_mode=choice(max,cat,lstm)' \
     'JKGCN_Cora.n_layer=range(2,10)'



     python3 train.py -m mlflow.runname=TwinGCN_CiteSeer_L246 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.n_layer=2' \
     'TwinGCN_CiteSeer.scope=choice(local,global)' \
     'TwinGCN_CiteSeer.kernel=choice(wdp,ad)' \
     'TwinGCN_CiteSeer.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinGCN_CiteSeer.activation=choice(ReLU,Identity)' \
     'TwinGCN_CiteSeer.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinGCN_CiteSeer_L246 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.n_layer=4' \
     'TwinGCN_CiteSeer.scope=choice(local,global)' \
     'TwinGCN_CiteSeer.kernel=choice(wdp,ad)' \
     'TwinGCN_CiteSeer.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinGCN_CiteSeer.activation=choice(ReLU,Identity)' \
     'TwinGCN_CiteSeer.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinGCN_CiteSeer_L246 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.n_layer=6' \
     'TwinGCN_CiteSeer.scope=choice(local,global)' \
     'TwinGCN_CiteSeer.kernel=choice(wdp,ad)' \
     'TwinGCN_CiteSeer.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinGCN_CiteSeer.activation=choice(ReLU,Identity)' \
     'TwinGCN_CiteSeer.self_loop=choice(True,False)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
