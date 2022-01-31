IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinSAGE_Products_L345 'key=TwinSAGE_Products' \
     'TwinSAGE_Products.n_layer=3' \
     'TwinSAGE_Products.scope=choice(local,global)' \
     'TwinSAGE_Products.kernel=choice(wdp,ad)' \
     'TwinSAGE_Products.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinSAGE_Products.activation=choice(ReLU,Identity)' \
     'TwinSAGE_Products.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinSAGE_Products_L345 'key=TwinSAGE_Products' \
     'TwinSAGE_Products.n_layer=4' \
     'TwinSAGE_Products.scope=choice(local,global)' \
     'TwinSAGE_Products.kernel=choice(wdp,ad)' \
     'TwinSAGE_Products.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinSAGE_Products.activation=choice(ReLU,Identity)' \
     'TwinSAGE_Products.self_loop=choice(True,False)'

     python3 train.py -m mlflow.runname=TwinSAGE_Products_L345 'key=TwinSAGE_Products' \
     'TwinSAGE_Products.n_layer=5' \
     'TwinSAGE_Products.scope=choice(local,global)' \
     'TwinSAGE_Products.kernel=choice(wdp,ad)' \
     'TwinSAGE_Products.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)' \
     'TwinSAGE_Products.activation=choice(ReLU,Identity)' \
     'TwinSAGE_Products.self_loop=choice(True,False)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
