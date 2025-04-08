#!/bin/bash

# parse args from command line
while getopts ":f:b:l:g:c:m:" opt; do
  case $opt in
    f) ft_stage="$OPTARG";;
    b) batch_size="$OPTARG";;
    l) learning_rate="$OPTARG";;
    g) gpu="$OPTARG";;
    c) cross_encoder="$OPTARG";;
    m) mode="$OPTARG";;
    *) echo "Invalid option -$OPTARG" >&2
       exit 1;;
  esac
done

# check if required arguments are provided
if [ -z "$ft_stage" ] || [ -z "$cross_encoder" ] || [ -z "$mode" ]; then
  echo "Usage: $0 -f <ft stage> -b <batch> -c <crossencoder model> -m <CL | KD> -l <lr> -g <gpu id>"
  exit 1
fi

LOGGER_NAME="$cross_encoder-ft$ft_stage-$mode"
CONFIG_FILE="configs/finetuning-$mode.yaml"
CONFIG_OPTIMIZER="configs/optimizer_s$ft_stage.yaml"

echo "mode: $mode"

if [ "$ft_stage" -eq 2 ]; then
  if test "$mode" = "CL"
  then
    previous_mode="KD"
  elif test "$mode" = "KD"
  then
    previous_mode="CL"
  else
    echo "Error: Invalid ft_stage value."
    exit 1
  fi
  CHECKPOINT_DIR="./../../data/models/$cross_encoder-ft1/$previous_mode/checkpoints"
fi


if [ "$ft_stage" -eq 1 ]; then
  if [ "$cross_encoder" = "electra" ]; then
    echo "setup MODEL_PATH to electra default checkpoint"
    MODEL_PATH="google/electra-base-discriminator"
  elif [ "$cross_encoder" = "bert" ]; then
    echo "setup MODEL_PATH to bert default checkpoint"
    MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2"
  elif [ "$cross_encoder" = "roberta" ]; then
    echo "setup MODEL_PATH to roberta default checkpoint"
    MODEL_PATH="FacebookAI/roberta-base"
  else
    echo "Error: Invalid crossencoder name."
    exit 1
  fi
elif [ "$ft_stage" -eq 2 ]; then
    MODEL_PATH="./../../data/models/$cross_encoder-ft1/$previous_mode/huggingface_checkpoint/"
else
    echo "Error: Invalid mode: it should be either \"CL\" or \"KD\"."
    exit 1
fi

echo "Starting finetuning's $ft_stage of mono$cross_encoder with mode=$mode"

CMD="CUDA_VISIBLE_DEVICES=$gpu python main.py fit \
    --config '$CONFIG_FILE' \
    --config '$CONFIG_OPTIMIZER' \
    --model.model_name_or_path='$MODEL_PATH' \
    --trainer.logger.init_args.name='$LOGGER_NAME'"

# add optional args 
if [ -n "$batch_size" ]; then
  CMD="$CMD --data.batch_size='$batch_size'"
fi

if [ -n "$learning_rate" ]; then
  CMD="$CMD --optimizer.lr='$learning_rate'"
fi

echo "checkpoint dir: $CHECKPOINT_DIR"

if [ "$ft_stage" -eq 2 ]; then
  # check if the directory exists
  if [ -d "$CHECKPOINT_DIR" ]; then
      # find .ckpt file
      ckpt_file=$(find "$CHECKPOINT_DIR" -type f -name "*.ckpt" -print -quit)

      if [ -n "$ckpt_file" ]; then
          echo "Found checkpoint file: $ckpt_file"
          CHECKPOINT_PATH="$ckpt_file"
          
      else
          echo "Error: No .ckpt file found in $CHECKPOINT_DIR"
          exit 1
      fi
  else
      echo "Error: Directory $CHECKPOINT_DIR does not exist"
      exit 1
  fi
fi

echo "**********\n**********\n**********\nRunning command CMD=$CMD **********\n**********\n**********\n"
# run the command
eval $CMD

# Example: nohup ./ft-monoencoder-CL-KD.sh -f 2 -c roberta -g 3 -m KD -l 1e-8 > finetuning-monoRoberta-s2-KD.out &
# Example: nohup ./ft-monoencoder-CL-KD.sh -f 2 -c roberta -g 3 -m CL -l 1e-8 > finetuning-monoRoBERTa-s2-CL.out &
# Example: nohup ./ft-monoencoder-CL-KD.sh -f 2 -c electra -g 2 -m CL -l 1e-8 > finetuning-monoElectra-s2-CL.out &