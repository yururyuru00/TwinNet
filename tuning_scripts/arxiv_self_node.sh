IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_Arxiv_tuning 'key=TwinGCN_Arxiv' \
     'TwinGCN_Arxiv.n_layer=range(2,7)' \
     'TwinGCN_Arxiv.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Arxiv.self_loop=False' \
     'TwinGCN_Arxiv.scope=local' \
     'TwinGCN_Arxiv.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Arxiv_tuning 'key=TwinGCN_Arxiv' \
     'TwinGCN_Arxiv.n_layer=range(2,7)' \
     'TwinGCN_Arxiv.dropout=choice(0.,0.2,0.5)' \
     'TwinGCN_Arxiv.learning_rate=choice(0.01,0.005,0.001)' \
     'TwinGCN_Arxiv.weight_decay=choice(0,1E-4,1E-3)' \
     'TwinGCN_Arxiv.self_loop=False' \
     'TwinGCN_Arxiv.scope=global' \
     'TwinGCN_Arxiv.temparature=choice(-0.1,-0.5,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done