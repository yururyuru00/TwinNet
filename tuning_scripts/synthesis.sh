IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_Synthesis_h0.0_tuning 'key=TwinGCN_Synthesis' \
     'TwinGCN_Synthesis.homophily=0.0' \
     'TwinGCN_Synthesis.n_layer=range(2,5)' \
     'TwinGCN_Synthesis.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_Synthesis.learning_rate=choice(0.05,0.01)' \
     'TwinGCN_Synthesis.weight_decay=choice(5e-5,5e-4,5e-3)' \
     'TwinGCN_Synthesis.self_loop=choice(True, False)' \
     'TwinGCN_Synthesis.scope=local' \
     'TwinGCN_Synthesis.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Synthesis_h0.0_tuning 'key=TwinGCN_Synthesis' \
     'TwinGCN_Synthesis.homophily=0.0' \
     'TwinGCN_Synthesis.n_layer=range(2,5)' \
     'TwinGCN_Synthesis.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_Synthesis.learning_rate=choice(0.05,0.01)' \
     'TwinGCN_Synthesis.weight_decay=choice(5e-5,5e-4,5e-3)' \
     'TwinGCN_Synthesis.self_loop=choice(True, False)' \
     'TwinGCN_Synthesis.scope=global' \
     'TwinGCN_Synthesis.temparature=choice(-0.1,-0.5,-1)'


     python3 train.py -m mlflow.runname=TwinGCN_Synthesis_h0.2_tuning 'key=TwinGCN_Synthesis' \
     'TwinGCN_Synthesis.homophily=0.2' \
     'TwinGCN_Synthesis.n_layer=range(2,5)' \
     'TwinGCN_Synthesis.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_Synthesis.learning_rate=choice(0.05,0.01)' \
     'TwinGCN_Synthesis.weight_decay=choice(5e-5,5e-4,5e-3)' \
     'TwinGCN_Synthesis.self_loop=choice(True, False)' \
     'TwinGCN_Synthesis.scope=local' \
     'TwinGCN_Synthesis.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Synthesis_h0.2_tuning 'key=TwinGCN_Synthesis' \
     'TwinGCN_Synthesis.homophily=0.2' \
     'TwinGCN_Synthesis.n_layer=range(2,5)' \
     'TwinGCN_Synthesis.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_Synthesis.learning_rate=choice(0.05,0.01)' \
     'TwinGCN_Synthesis.weight_decay=choice(5e-5,5e-4,5e-3)' \
     'TwinGCN_Synthesis.self_loop=choice(True, False)' \
     'TwinGCN_Synthesis.scope=global' \
     'TwinGCN_Synthesis.temparature=choice(-0.1,-0.5,-1)'


     python3 train.py -m mlflow.runname=TwinGCN_Synthesis_h0.7_tuning 'key=TwinGCN_Synthesis' \
     'TwinGCN_Synthesis.homophily=0.7' \
     'TwinGCN_Synthesis.n_layer=range(2,5)' \
     'TwinGCN_Synthesis.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_Synthesis.learning_rate=choice(0.05,0.01)' \
     'TwinGCN_Synthesis.weight_decay=choice(5e-5,5e-4,5e-3)' \
     'TwinGCN_Synthesis.self_loop=choice(True, False)' \
     'TwinGCN_Synthesis.scope=local' \
     'TwinGCN_Synthesis.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Synthesis_h0.7_tuning 'key=TwinGCN_Synthesis' \
     'TwinGCN_Synthesis.homophily=0.7' \
     'TwinGCN_Synthesis.n_layer=range(2,5)' \
     'TwinGCN_Synthesis.dropout=choice(0.,0.2,0.4)' \
     'TwinGCN_Synthesis.learning_rate=choice(0.05,0.01)' \
     'TwinGCN_Synthesis.weight_decay=choice(5e-5,5e-4,5e-3)' \
     'TwinGCN_Synthesis.self_loop=choice(True, False)' \
     'TwinGCN_Synthesis.scope=global' \
     'TwinGCN_Synthesis.temparature=choice(-0.1,-0.5,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done