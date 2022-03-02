IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=for_paper 'key=GCN_Cora' \
     'GCN_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=GAT_Cora' \
     'GAT_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKGAT_Cora' \
     'JKGAT_Cora.jk_mode=choice(max,cat,lstm)' \
     'JKGAT_Cora.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinGAT_Cora' \
     'TwinGAT_Cora.n_layer=range(2,10)'


     python3 train.py -m mlflow.runname=for_paper 'key=GCN_CiteSeer' \
     'GCN_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=GAT_CiteSeer' \
     'GAT_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKGAT_CiteSeer' \
     'JKGAT_CiteSeer.jk_mode=choice(max,cat,lstm)' \
     'JKGAT_CiteSeer.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinGAT_CiteSeer' \
     'TwinGAT_CiteSeer.n_layer=range(2,10)'


     python3 train.py -m mlflow.runname=for_paper 'key=GCN_PubMed' \
     'GCN_PubMed.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=GAT_PubMed' \
     'GAT_PubMed.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKGAT_PubMed' \
     'JKGAT_PubMed.jk_mode=choice(max,cat,lstm)' \
     'JKGAT_PubMed.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.n_layer=range(2,10)'


     python3 train.py -m mlflow.runname=for_paper 'key=GCN_PPIinduct' \
     'GCN_PPIinduct.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=GAT_PPIinduct' \
     'GAT_PPIinduct.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=JKGAT_PPIinduct' \
     'JKGAT_PPIinduct.jk_mode=choice(max,cat,lstm)' \
     'JKGAT_PPIinduct.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=for_paper 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.n_layer=range(2,10)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
