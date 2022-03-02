IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=for_paper 'key=GCN_PPI' \
     'GCN_PPI.n_layer=range(2,8)'

     python3 train.py -m mlflow.runname=for_paper 'key=SAGE_PPI' \
     'SAGE_PPI.n_layer=range(2,8)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKSAGE_PPI' \
     'JKSAGE_PPI.jk_mode=choice(max,cat,lstm)' \
     'JKSAGE_PPI.n_layer=range(2,8)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinSAGE_PPI' \
     'TwinSAGE_PPI.n_layer=range(2,8)'


     python3 train.py -m mlflow.runname=for_paper 'key=GCN_Reddit' \
     'GCN_Reddit.n_layer=range(2,4)'

     python3 train.py -m mlflow.runname=for_paper 'key=SAGE_Reddit' \
     'SAGE_Reddit.n_layer=range(2,4)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKSAGE_Reddit' \
     'JKSAGE_Reddit.jk_mode=choice(max,cat,lstm)' \
     'JKSAGE_Reddit.n_layer=range(2,4)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinSAGE_Reddit' \
     'TwinSAGE_Reddit.n_layer=range(2,4)'


     python3 train.py -m mlflow.runname=for_paper 'key=GCN_Arxiv' \
     'GCN_Arxiv.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=SAGE_Arxiv' \
     'SAGE_Arxiv.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKSAGE_Arxiv' \
     'JKSAGE_Arxiv.jk_mode=choice(max,cat,lstm)' \
     'JKSAGE_Arxiv.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinSAGE_Arxiv' \
     'TwinSAGE_Arxiv.n_layer=range(2,10)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
