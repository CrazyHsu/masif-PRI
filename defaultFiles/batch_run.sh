#!/bin/bash

#i=1
#while read p; do
#  if [ $(( i % 1001 )) == ${SLURM_ARRAY_TASK_ID} ]; then
#    python masifPNI.py masifPNI-site dataprep -l $p --config defaultFiles/test.cfg -n 1 -overwrite
#  fi
#  i=$((i+1))
#done < $1

cat /dev/null > batch_run.lst
while read p; do
  echo "python masifPNI.py masifPNI-site dataprep -l $p --config defaultFiles/test.cfg -overwrite --nobatchRun" >> batch_run.lst
done < $1

#parallel -j $2 < batch_run.lst
