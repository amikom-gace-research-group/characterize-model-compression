#!/bin/bash

source ~/garuda_env/bin/activate

module_name="mobilenet_v3"
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)
sparsities=(
	"0.25"
	"0.50"
	"0.75"
)

for model in "${models[@]}"
do
	for sparsity in "${sparsities[@]}"
	do
		log_output="./results/pruning/cpu_tflite/${model}_${sparsity}_sparsity.txt"

		for i in {1..5} 
		do
			general_options=(
				--module_name ${module_name}
				--model_name ${model}
				--run_num "${i}"
				--target_sparsity ${sparsity}
			)

			pruning_options=(
				--tuner CPU
			)

			# Run pruning in python
			python3 pyscripts/pruning/prune.py "${general_options[@]}" "${pruning_options[@]}" >> $log_output

			# Quantization
			# python3 pyscripts/pruning/ptq_tflite_pruned.py "${general_options[@]}" >> $log_output

			printf "\n-----------------------------------\n" >> $log_output
		done
	done
done
