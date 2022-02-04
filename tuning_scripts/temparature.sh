IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=temparature 'key=TwinGCN_Cora' \
     'TwinGCN_Cora.scope=local' \
     'TwinGCN_Cora.temparature=choice(0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)'

     python3 train.py -m mlflow.runname=temparature 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.scope=local' \
     'TwinGCN_CiteSeer.temparature=choice(0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)'

     python3 train.py -m mlflow.runname=temparature 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.scope=local' \
     'TwinGAT_PubMed.temparature=choice(0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)'

     python3 train.py -m mlflow.runname=temparature 'key=TwinSAGE_PPI' \
     'TwinSAGE_PPI.scope=local' \
     'TwinSAGE_PPI.temparature=choice(0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)'

     python3 train.py -m mlflow.runname=temparature 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.scope=global' \
     'TwinGAT_PPIinduct.temparature=choice(-0.001,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1)'

     python3 train.py -m mlflow.runname=temparature 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.scope=global' \
     'TwinSAGE_Reddit.temparature=choice(-0.001,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
