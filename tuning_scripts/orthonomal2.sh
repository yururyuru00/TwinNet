IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinSAGE_Arxiv' \
     'TwinSAGE_Arxiv.n_layer=range(2,10)' \
     'TwinSAGE_Arxiv.coef_orthonomal=100'

     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinSAGE_PPI' \
     'TwinSAGE_PPI.n_layer=range(2,8)' \
     'TwinSAGE_PPI.coef_orthonomal=100'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
