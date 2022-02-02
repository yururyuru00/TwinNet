IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=range(2,7)'

     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=SAGE_Reddit' \
     'SAGE_Reddit.n_layer=2'

     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=JKSAGE_Reddit' \
     'JKSAGE_Reddit.jk_mode=choice(max,cat,lstm)' \
     'JKSAGE_Reddit.n_layer=2'

     
     python3 train.py -m mlflow.runname=TwinRGCN_MAG_tuning 'key=TwinRGCN_MAG' \
     'TwinRGCN_MAG.n_layer=choice(2,3)' \
     'TwinRGCN_MAG.scope=local' \
     'TwinRGCN_MAG.kernel=choice(wdp,sdp,dp)' \
     'TwinRGCN_MAG.temparature=choice(0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinRGCN_MAG_tuning 'key=TwinRGCN_MAG' \
     'TwinRGCN_MAG.n_layer=choice(2,3)' \
     'TwinRGCN_MAG.scope=global' \
     'TwinRGCN_MAG.kernel=choice(wdp,sdp,dp)' \
     'TwinRGCN_MAG.temparature=choice(-0.1,-0.5,-1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
