
########################< DVC USE CASE >########################
$dvc init                     #Initialize the repository

#Add a remote repository
$dvc remote add -d myremote /tmp/dvc-storage
$dvc add data/data.csv        #Track data file
$git add .
$git commit -m 'data added'
$dvc push                     #Push from local to remote repository
$dvc pull data/data.xml.dvc   #Data pulling

#MODEL RUN
$dvc run -f evaluate.dvc \
          -d src/evaluate.py -d model.pkl -d data/features \
          -M auc.metric \
          python src/evaluate.py model.pkl \
                 data/features auc.metric
