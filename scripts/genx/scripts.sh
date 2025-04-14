#!/bin/bash
NUM_PROCESSES=5  # set to the number of parallel processes to use
DATA_DIR=/path/to/input
DATASET=gen4 ## gen1 or gen4 or DSEC
DT=(5 10 20 50 100)  # Different duration values

for dt in "${DT[@]}"; do
    DEST_DIR="/path/to/output/${DATASET}_preprocessed/dt_${dt}"  # Dynamic output directory
    CONFIG_DURATION="conf_preprocess/extraction/duration_${dt}.yaml"  # Dynamic YAML file

    echo "Processing with dt=${dt}, saving to ${DEST_DIR}, using config ${CONFIG_DURATION}"

    python3 preprocess_dataset.py "${DATA_DIR}" "${DEST_DIR}" \
        conf_preprocess/representation/stacked_hist.yaml \
        "${CONFIG_DURATION}" \
        conf_preprocess/filter_${DATASET}.yaml \
        -ds ${DATASET} -np "${NUM_PROCESSES}"
done
