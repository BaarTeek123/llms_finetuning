#!/bin/bash


datasets=("mnli" "qnli" "qqp" "sst2")
file_names=("ia3.py" "full_fine_tuning.py" "lora.py" "prefix.py" "prefix_lora.py" "top_layer.py" "soft_prompt.py" "soft_prompt_lora.py" "dp_full_ft.py" "dp_top_layer.py" "dp_lora.py" "dp_prefix.py" "dp_soft_prompt.py")
file_names=("prefix.py" "top_layer.py")

# Loop through each Python script
for file in "${file_names[@]}"; do
  # Check if the script file exists
  if [[ -f "$file" ]]; then
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      echo "Running $file with dataset $dataset"
      python "$file" "$dataset"
    done
  else
    echo "Error: $file not found"
  fi
done



file_names=("dp_full_ft.py" "dp_top_layer.py" "dp_lora.py" "dp_prefix.py" "dp_soft_prompt.py")
epsilons=("8")


# Loop through each Python script
for file in "${file_names[@]}"; do
  # Check if the script file exists
  if [[ -f "$file" ]]; then
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
      for eps in "${epsilons[@]}"; do
        echo "Running $file with dataset $dataset --epsilon $eps"
        python "$file" "$dataset" epsilon "$eps"
      done
    done
  else
    echo "Error: $file not found"
  fi
done
