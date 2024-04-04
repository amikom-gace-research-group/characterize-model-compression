#!/bin/bash

source ~/garuda_env/bin/activate

# ===== GENERAL CONFIG =====
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

sparsities=(
	"0.25"
	"0.50"
	"0.75"
)

# ===== LATENCY TESTS =====
for model in "${models[@]}"
do
	for sparsity in "${sparsities[@]}"
	do
		log_file_loc="./results/pruning/latencies/tflite/${model}_${sparsity}.txt"
		quantized_path="./generated/pruned_tflite/${model}_${sparsity}_sparsity/run_2/pruned_quantized.tflite"
		benchmarker="$HOME/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"

		for i in {1..10}
		do
			$benchmarker --graph=${quantized_path} --warmup_runs=10 --num_runs=1000 >> $log_file_loc
			printf "\n-------------------------\n" >> $log_file_loc
		done
	done
done
