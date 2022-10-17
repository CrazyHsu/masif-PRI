#!/bin/bash

#i=1
#while read p; do
#  if [ $(( i % 1001 )) == ${SLURM_ARRAY_TASK_ID} ]; then
#    python masifPNI.py masifPNI-site dataprep -l $p --config defaultFiles/test.cfg -n 1 -overwrite
#  fi
#  i=$((i+1))
#done < $1

cat /dev/null > $2
while read p; do
  echo "python masifPNI.py masifPNI-site dataprep -l $p --config defaultFiles/test.cfg -overwrite --nobatchRun" >> $2
done < $1

#parallel -j $2 < batch_run.lst
