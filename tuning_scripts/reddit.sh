IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=4' \
     'TwinSAGE_Reddit.dropout=0.3' \
     'TwinSAGE_Reddit.scope=global' \
     'TwinSAGE_Reddit.temparature=-0.1'  

     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=4' \
     'TwinSAGE_Reddit.dropout=0.' \
     'TwinSAGE_Reddit.scope=global' \
     'TwinSAGE_Reddit.temparature=choice(-1,-0.5,-0.1)'  

     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=5' \
     'TwinSAGE_Reddit.dropout=choice(0.6,0.3,0.)' \
     'TwinSAGE_Reddit.scope=global' \
     'TwinSAGE_Reddit.temparature=choice(-1,-0.5,-0.1)'

     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=choice(3,4,5)' \
     'TwinSAGE_Reddit.dropout=choice(0.6,0.3,0.)' \
     'TwinSAGE_Reddit.scope=local' \
     'TwinSAGE_Reddit.temparature=choice(1,0.5,0.1)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
