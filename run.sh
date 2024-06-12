#!/bin/bash
# Script: run_all_datasets.sh

datasets=("mnli" "qnli" "qqp" "sst2")
file_names=("ia3.py" "full_fine_tuning.py" "lora.py" "prefix_lora.py" "top_layer.py")



# Loop through each Python script
for file in "${file_names[@]}"; do
  # Check if the script file exists
  if [[ -f "$file" ]]; then
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      echo "Running $file with --dataset $dataset"
      python "$file" "$dataset"
    done
  else
    echo "Error: $file not found"
  fi
done
