IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_Cornell_tuning 'key=TwinGCN_Cornell' \
     'TwinGCN_Cornell.n_layer=range(2,5)' \
     'TwinGCN_Cornell.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Cornell.self_loop=choice(True, False)' \
     'TwinGCN_Cornell.scope=local' \
     'TwinGCN_Cornell.temparature=choice(0.1,0.3,0.5,0.7,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Cornell_tuning 'key=TwinGCN_Cornell' \
     'TwinGCN_Cornell.n_layer=range(2,5)' \
     'TwinGCN_Cornell.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Cornell.self_loop=choice(True, False)' \
     'TwinGCN_Cornell.scope=global' \
     'TwinGCN_Cornell.temparature=choice(-0.1,-0.3,-0.5,-0.7,-1)'


     python3 train.py -m mlflow.runname=TwinGCN_Texas_tuning 'key=TwinGCN_Texas' \
     'TwinGCN_Texas.n_layer=range(2,5)' \
     'TwinGCN_Texas.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Texas.self_loop=choice(True, False)' \
     'TwinGCN_Texas.scope=local' \
     'TwinGCN_Texas.temparature=choice(0.1,0.3,0.5,0.7,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Texas_tuning 'key=TwinGCN_Texas' \
     'TwinGCN_Texas.n_layer=range(2,5)' \
     'TwinGCN_Texas.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Texas.self_loop=choice(True, False)' \
     'TwinGCN_Texas.scope=global' \
     'TwinGCN_Texas.temparature=choice(-0.1,-0.3,-0.5,-0.7,-1)'


     python3 train.py -m mlflow.runname=TwinGCN_Wisconsin_tuning 'key=TwinGCN_Wisconsin' \
     'TwinGCN_Wisconsin.n_layer=range(2,5)' \
     'TwinGCN_Wisconsin.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Wisconsin.self_loop=choice(True, False)' \
     'TwinGCN_Wisconsin.scope=local' \
     'TwinGCN_Wisconsin.temparature=choice(0.1,0.3,0.5,0.7,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Wisconsin_tuning 'key=TwinGCN_Wisconsin' \
     'TwinGCN_Wisconsin.n_layer=range(2,5)' \
     'TwinGCN_Wisconsin.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Wisconsin.self_loop=choice(True, False)' \
     'TwinGCN_Wisconsin.scope=global' \
     'TwinGCN_Wisconsin.temparature=choice(-0.1,-0.3,-0.5,-0.7,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done