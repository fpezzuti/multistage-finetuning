#!/bin/bash

# parse args from command line
while getopts ":f:b:g:d:c:m:" opt; do
  case $opt in
    f) ft_stage="$OPTARG";;
    b) batch_size="$OPTARG";;
    g) gpu="$OPTARG";;
    d) test="$OPTARG";;
    c) cross_encoder="$OPTARG";;
    m) mode="$OPTARG";;
    *) echo "Invalid option -$OPTARG" >&2
       exit 1;;
  esac
done

# check if required arguments are provided
if [ -z "$ft_stage" ] || [ -z "$cross_encoder" ] || [ -z "$gpu" ] || [ -z "$test" ] || [ -z "$mode" ]; then
  echo "Usage: $0 -f <ft stage> -c <crossencoder model> -m <CL | KD> -b <batch> -d [indomain or outdomain] -g <gpu id>"
  exit 1
fi


MODEL_PATH="./../../data/models/$cross_encoder-ft$ft_stage/$mode/huggingface_checkpoint/"
CHECKPOINT_DIR="./../../data/models/$cross_encoder-ft$ft_stage/$mode/checkpoints"
CONFIG_INFERENCE="configs/inference-$test.yaml"

echo "Initialising inference of mono$cross_encoder of stage $ft_stage."

CMD="CUDA_VISIBLE_DEVICES=$gpu python main.py test \
    --config '$CONFIG_INFERENCE' \
    --model.model_name_or_path='$MODEL_PATH'"

# add optional args 
if [ -n "$batch_size" ]; then
  CMD="$CMD --data.batch_size='$batch_size'"
fi

if [ "$ft_stage" -eq 2 ]; then
  # check if the directory exists
  if [ -d "$CHECKPOINT_DIR" ]; then
      # find .ckpt file
      ckpt_file=$(find "$CHECKPOINT_DIR" -type f -name "*.ckpt" -print -quit)

      if [ -n "$ckpt_file" ]; then
          echo "Found checkpoint file: $ckpt_file"
          CHECKPOINT_PATH="$ckpt_file"
          CMD="$CMD --ckpt_path='$CHECKPOINT_PATH'" 
      else
          echo "Error: No .ckpt file found in $CHECKPOINT_DIR"
          exit 1
      fi
  else
      echo "Error: Directory $checkpoint_dir does not exist"
      exit 1
  fi
fi

echo $CMD
# run the command
eval $CMD

# Example: ./inference-monoencoder.sh -f 1 -c electra -m CL -d indomain -g 2