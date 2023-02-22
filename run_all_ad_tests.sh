#!/bin/bash

#METHODS="GrangerPW GrangerMV TCDF PCMCIParCorr PCMCICMIknn tsFCI VarLiNGAM TiMINO Dynotears" GrangerVM relies upon matlab for some reason
#METHODS="GrangerPW TCDF PCMCIParCorr PCMCICMIknn tsFCI VarLiNGAM TiMINO Dynotears" Apparently some of these methods were implemented without a care for multithread processing
METHODS="GrangerPW GrangerMV TCDF PCMCIParCorr tsFCI VarLiNGAM TiMINO Dynotears NAVARMLP CDNOD Random"
DATASETS="lyft highd synthetic"
VARIABLES="acceleration velocity"
PROCESSOR_COUNT=1
MAX_TIME_LAG=25
SIG_LEVEL=0.2

if [ "${SKIP_TO_METHOD}" != "" ]; then
  NEW_METHODS=""
  for METHOD in ${METHODS}; do
    if [[ "${NEW_METHODS}" != "" || "${METHOD}" == "${SKIP_TO_METHOD}" ]]; then
      NEW_METHODS="${NEW_METHODS} ${METHOD}"
    fi
  done
  METHODS="${NEW_METHODS}"
  unset NEW_METHODS
fi

if [ "${OVERRIDE_METHODS}" != "" ]; then
  METHODS="${OVERRIDE_METHODS}"
fi

if [ "${OVERRIDE_DATASETS}" != "" ]; then
  DATASETS="${OVERRIDE_DATASETS}"
fi

if [ "${OVERRIDE_VARIABLES}" != "" ]; then
  VARIABLES="${OVERRIDE_VARIABLES}"
fi

for METHOD in ${METHODS}; do
  for DATASET in ${DATASETS}; do
    for VARIABLE in ${VARIABLES}; do
      echo "Running the ${METHOD} method on the ${VARIABLE} variables of agents from the ${DATASET} dataset"
      ./test_ad.py ${METHOD} ${DATASET} ${VARIABLE} --processor-count ${PROCESSOR_COUNT} --verbose --max-time-lag ${MAX_TIME_LAG} --sig-level ${SIG_LEVEL}
      if [ $? -ne 0 ]; then
        echo "Error encountered while running the ${METHOD} method on the ${VARIABLE} variables of agents from the ${DATASET} dataset"
        exit 1
      fi
    done
  done
done
