IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=for_paper_orthonomal2 'key=TwinGAT_PPIinduct' \
     'TwinGAT_PPIinduct.n_layer=range(2,10)' \
     'TwinGAT_PPIinduct.coef_orthonomal=100'
     ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
