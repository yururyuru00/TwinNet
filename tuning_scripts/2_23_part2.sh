IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=2_23 'key=GCN_PPI' \
     'GCN_PPI.n_layer=range(2,8)'

     python3 train.py -m mlflow.runname=2_23 'key=SAGE_PPI' \
     'SAGE_PPI.n_layer=range(2,8)'

     python3 train.py -m mlflow.runname=2_23 'key=JKSAGE_PPI' \
     'JKSAGE_PPI.jk_mode=choice(max,cat,lstm)' \
     'JKSAGE_PPI.n_layer=range(2,8)'

     python3 train.py -m mlflow.runname=2_23 'key=TwinSAGE_PPI' \
     'TwinSAGE_PPI.n_layer=range(2,8)'


     python3 train.py -m mlflow.runname=2_23 'key=GCN_Reddit' \
     'GCN_Reddit.n_layer=range(2,4)'

     python3 train.py -m mlflow.runname=2_23 'key=SAGE_Reddit' \
     'SAGE_Reddit.n_layer=range(2,4)'

     python3 train.py -m mlflow.runname=2_23 'key=JKSAGE_Reddit' \
     'JKSAGE_Reddit.jk_mode=choice(max,cat,lstm)' \
     'JKSAGE_Reddit.n_layer=range(2,4)'

     python3 train.py -m mlflow.runname=2_23 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=range(2,4)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
