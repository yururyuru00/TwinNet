IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_SynCora_h0.0_tuning 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.homophily=0.0' \
     'TwinGCN_SynCora.n_layer=range(2,5)' \
     'TwinGCN_SynCora.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_SynCora.learning_rate=choice(0.05,0.01,0.001)' \
     'TwinGCN_SynCora.weight_decay=choice(1e-3, 5e-4, 1e-4)' \
     'TwinGCN_SynCora.self_loop=choice(True, False)' \
     'TwinGCN_SynCora.scope=local' \
     'TwinGCN_SynCora.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_SynCora_h0.0_tuning 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.homophily=0.0' \
     'TwinGCN_SynCora.n_layer=range(2,5)' \
     'TwinGCN_SynCora.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_SynCora.learning_rate=choice(0.05,0.01,0.001)' \
     'TwinGCN_SynCora.weight_decay=choice(1e-3, 5e-4, 1e-4)' \
     'TwinGCN_SynCora.self_loop=choice(True, False)' \
     'TwinGCN_SynCora.scope=global' \
     'TwinGCN_SynCora.temparature=choice(-0.1,-0.5,-1)'


     python3 train.py -m mlflow.runname=TwinGCN_SynCora_h0.2_tuning 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.homophily=0.2' \
     'TwinGCN_SynCora.n_layer=range(2,5)' \
     'TwinGCN_SynCora.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_SynCora.learning_rate=choice(0.05,0.01,0.001)' \
     'TwinGCN_SynCora.weight_decay=choice(1e-3, 5e-4, 1e-4)' \
     'TwinGCN_SynCora.self_loop=choice(True, False)' \
     'TwinGCN_SynCora.scope=local' \
     'TwinGCN_SynCora.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_SynCora_h0.2_tuning 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.homophily=0.2' \
     'TwinGCN_SynCora.n_layer=range(2,5)' \
     'TwinGCN_SynCora.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_SynCora.learning_rate=choice(0.05,0.01,0.001)' \
     'TwinGCN_SynCora.weight_decay=choice(1e-3, 5e-4, 1e-4)' \
     'TwinGCN_SynCora.self_loop=choice(True, False)' \
     'TwinGCN_SynCora.scope=global' \
     'TwinGCN_SynCora.temparature=choice(-0.1,-0.5,-1)'


     python3 train.py -m mlflow.runname=TwinGCN_SynCora_h0.8_tuning 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.homophily=0.8' \
     'TwinGCN_SynCora.n_layer=range(2,5)' \
     'TwinGCN_SynCora.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_SynCora.learning_rate=choice(0.05,0.01,0.001)' \
     'TwinGCN_SynCora.weight_decay=choice(1e-3, 5e-4, 1e-4)' \
     'TwinGCN_SynCora.self_loop=choice(True, False)' \
     'TwinGCN_SynCora.scope=local' \
     'TwinGCN_SynCora.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_SynCora_h0.8_tuning 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.homophily=0.8' \
     'TwinGCN_SynCora.n_layer=range(2,5)' \
     'TwinGCN_SynCora.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_SynCora.learning_rate=choice(0.05,0.01,0.001)' \
     'TwinGCN_SynCora.weight_decay=choice(1e-3, 5e-4, 1e-4)' \
     'TwinGCN_SynCora.self_loop=choice(True, False)' \
     'TwinGCN_SynCora.scope=global' \
     'TwinGCN_SynCora.temparature=choice(-0.1,-0.5,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done