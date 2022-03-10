IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_Cora' \
     'TwinGAT_Cora.n_layer=range(2,10)' \
     'TwinGAT_Cora.coef_orthonomal=100'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_CiteSeer' \
     'TwinGAT_CiteSeer.n_layer=range(2,10)' \
     'TwinGAT_CiteSeer.coef_orthonomal=100'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.n_layer=range(2,10)' \
     'TwinGAT_PubMed.coef_orthonomal=100'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.n_layer=range(2,10)' \
     'TwinGAT_PPIinduct.coef_orthonomal=100'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
