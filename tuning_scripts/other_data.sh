IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=TwinGCN_Arxiv_L36 'key=TwinGCN_Arxiv' \
     'TwinGCN_Arxiv.n_tri=2' \
     'TwinGCN_Arxiv.n_layer=3' \
     'TwinGCN_Arxiv.scope=choice(local,global)' \
     'TwinGCN_Arxiv.kernel=choice(wdp,ad)' \
     'TwinGCN_Arxiv.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)'

     python3 train.py -m mlflow.runname=TwinGCN_Arxiv_L36 'key=TwinGCN_Arxiv' \
     'TwinGCN_Arxiv.n_tri=2' \
     'TwinGCN_Arxiv.n_layer=6' \
     'TwinGCN_Arxiv.scope=choice(local,global)' \
     'TwinGCN_Arxiv.kernel=choice(wdp,ad)' \
     'TwinGCN_Arxiv.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)'


     python3 train.py -m mlflow.runname=TwinGAT_PPIinduct_L5 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.n_tri=3' \
     'TwinGAT_PPIinduct.n_layer=5' \
     'TwinGAT_PPIinduct.scope=choice(local,global)' \
     'TwinGAT_PPIinduct.kernel=choice(wdp,ad)' \
     'TwinGAT_PPIinduct.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)'


     python3 train.py -m mlflow.runname=TwinSAGE_PPI_L7 'key=TwinSAGE_PPI' \
     'TwinSAGE_PPI.n_tri=1' \
     'TwinSAGE_PPI.n_layer=7' \
     'TwinSAGE_PPI.scope=choice(local,global)' \
     'TwinSAGE_PPI.kernel=choice(wdp,ad)' \
     'TwinSAGE_PPI.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)'


     python3 train.py -m mlflow.runname=TwinSAGE_Reddit_L3 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_tri=2' \
     'TwinSAGE_Reddit.n_layer=3' \
     'TwinSAGE_Reddit.scope=choice(local,global)' \
     'TwinSAGE_Reddit.kernel=choice(wdp,ad)' \
     'TwinSAGE_Reddit.temparature=choice(-1,-0.5,-0.1,0.1,0.5,1)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
