IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_Cora' \
     'TwinGAT_Cora.coef_orthonomal=choice(0, 100)'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_CiteSeer' \
     'TwinGAT_CiteSeer.coef_orthonomal=choice(0, 100)'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_PubMed' \
     'TwinGAT_PubMed.coef_orthonomal=choice(0, 100)'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.coef_orthonomal=choice(0, 100)'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinSAGE_Arxiv' \
     'TwinSAGE_Arxiv.coef_orthonomal=choice(0, 100)'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinSAGE_PPI' \
     'TwinSAGE_PPI.coef_orthonomal=choice(0, 100)'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
