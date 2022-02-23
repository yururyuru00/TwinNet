IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_Wisconsin_fulltuning 'key=TwinGCN_Wisconsin' \
     'TwinGCN_Wisconsin.n_layer=4' \
     'TwinGCN_Wisconsin.dropout=0.5' \
     'TwinGCN_Wisconsin.learning_rate=choice(0.005,0.001)' \
     'TwinGCN_Wisconsin.weight_decay=choice(0.,5e-6,5e-5,5e-4,5e-3)' \
     'TwinGCN_Wisconsin.self_loop=choice(True, False)' \
     'TwinGCN_Wisconsin.scope=global' \
     'TwinGCN_Wisconsin.temparature=choice(-0.1,-0.3,-0.5,-0.7,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done

