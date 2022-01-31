IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=GCN_CiteSeer_L2-9 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_CiteSeer_L2-9 'key=GCN_CiteSeer' \
     'GCN_CiteSeer.skip_connection=choice(vanilla,res,dense,highway)' \
     'GCN_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_CiteSeer_L2-9 'key=JKGCN_CiteSeer' \
     'JKGCN_CiteSeer.jk_mode=choice(max,cat,lstm)' \
     'JKGCN_CiteSeer.n_layer=range(2,10)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
