MODEL_NAME="Deepseek-R1" # replace with your actual model in model_configs in infer/models/__init__.py, and fill in the corresponding key of that dict.
SPLIT="operation_research formal_language physics zebra logic_calculation cipher_and_code puzzle_and_code number_calculation"
MODE="zero-shot"
CODE_MODE="noncode" # You can also select pot/agent/sandbox

output_dir="results/test_ds-r1"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
echo "Starting noncode inference for $MODEL_NAME..."


python infer/infer.py \
  --model_name $MODEL_NAME \
  --model $MODEL_NAME \
  --split $SPLIT \
  --mode $MODE \
  --code_mode $CODE_MODE \
  --output_dir $output_dir \
  --num_workers 128

FOLDER_NAME="ds-r1"
SOURCE_FOLDER=$output_dir
TARGET_FOLDER="eval/results/ds-r1-v1"
CSV_FILE=${TARGET_FOLDER}/ds-r1_noncode_evaluation.csv
MAX_WORKERS=8


if [ ! -d "$TARGET_FOLDER" ]; then
    mkdir -p "$TARGET_FOLDER"
fi

echo "Evaluating noncode results for $FOLDER_NAME"
echo "Source: $SOURCE_FOLDER"
echo "Target: $TARGET_FOLDER"
echo "CSV: $CSV_FILE"

python eval/eval.py \
  "$SOURCE_FOLDER" \
  "$TARGET_FOLDER" \
  "$CSV_FILE" \
  --use_llm_judge \
  --max_workers $MAX_WORKERS

echo "Started noncode evaluation for $FOLDER_NAME"
