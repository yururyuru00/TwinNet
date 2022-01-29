IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 train.py -m mlflow.runname=JKSAGE_PPI 'key=JKSAGE_PPI' \
     'JKSAGE_PPI.jk_mode=choice(max, cat, lstm)' \
     'JKSAGE_PPI.n_layer=range(2,10)'

     python3 train.py -m mlflow.runname=SAGE_PPI 'key=SAGE_PPI' \
     'SAGE_PPI.n_layer=range(2,10)'
    ")


for STR in ${ary[@]}
do
    eval "${STR}"
done
