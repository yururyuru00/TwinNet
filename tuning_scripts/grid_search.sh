IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=GAT_PubMed_L2-9 'key=TwinGAT_PubMed' \
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
