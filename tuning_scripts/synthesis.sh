IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=GCN_SynCora 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.self_loop=True' \
     'TwinGCN_SynCora.homophily=choice(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)'

     python3 train.py -m mlflow.runname=GCN_SynCora 'key=TwinGCN_SynCora' \
     'TwinGCN_SynCora.self_loop=False' \
     'TwinGCN_SynCora.homophily=choice(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)'

     python3 train.py -m mlflow.runname=GCN_SynCora 'key=GCN_SynCora' \
     'GCN_SynCora.homophily=choice(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done