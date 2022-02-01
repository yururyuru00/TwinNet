IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=GCN_Cora_L2-9 'key=TwinGCN_Cora' \
     'TwinGCN_Cora.temparature=choice(0.1,0.5,1)' \
     'TwinGCN_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_Cora_L2-9 'key=JKGCN_Cora' \
     'JKGCN_Cora.jk_mode=choice(max,cat,lstm)' \
     'JKGCN_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_Cora_L2-9 'key=GCN_Cora' \
     'GCN_Cora.skip_connection=choice(vanilla,res,dense,highway)' \
     'GCN_Cora.n_layer=range(2,10)'


     python3 train.py -m mlflow.runname=GCN_CiteSeer_L2-9 'key=TwinGCN_CiteSeer' \
     'TwinGCN_CiteSeer.temparature=choice(0.1,0.5,1)' \
     'TwinGCN_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_CiteSeer_L2-9 'key=JKGCN_CiteSeer' \
     'JKGCN_CiteSeer.jk_mode=choice(max,cat,lstm)' \
     'JKGCN_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GCN_CiteSeer_L2-9 'key=GCN_CiteSeer' \
     'GCN_CiteSeer.skip_connection=choice(vanilla,res,dense,highway)' \
     'GCN_CiteSeer.n_layer=range(2,10)'


     python3 train.py -m mlflow.runname=GAT_PubMed_L2-9 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.temparature=choice(0.1,0.5,1)' \
     'TwinGAT_PubMed.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GAT_PubMed_L2-9 'key=JKGAT_PubMed' \
     'JKGAT_PubMed.jk_mode=choice(max,cat,lstm)' \
     'JKGAT_PubMed.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=GAT_PubMed_L2-9 'key=GAT_PubMed' \
     'GAT_PubMed.skip_connection=choice(vanilla,res,dense,highway)' \
     'GAT_PubMed.n_layer=range(2,10)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
