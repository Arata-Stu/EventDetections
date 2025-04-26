MODEL=rvt ## rvt, rvt_ssm, yolox, yolox_lstm
SIZE=tiny ## tiny small base
DATASET=gen4 ## gen1, gen4, VGA (640*480 ev cam)
EV_REPR_NAME='stacked_histogram_dt=5_nbins=1'
INPUT_PATH=/media/arata-22/AT_SSD/dataset/gen4_preprocessed_bins_1/dt_5 # preprocessed dataset
OUTPUT_VIDEO=dt_5.mp4
CKPT_PATH=/home/arata-22/Downloads/EventDetections/rvt_gen4_bins_1/dt_5_best.ckpt
BINS=1
CHANNEL=2
python3 create_video.py \
model=${MODEL} +model/${MODEL}=${SIZE}.yaml model.backbone.input_channels=${CHANNEL} \
dataset=${DATASET} dataset.path=${INPUT_PATH} dataset.ev_repr_name="'${EV_REPR_NAME}'" \
output_path=${OUTPUT_VIDEO} gt=False pred=False num_sequence=1 ckpt_path=${CKPT_PATH} \