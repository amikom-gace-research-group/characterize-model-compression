#!/bin/bash

source ~/garuda_env/bin/activate

# ===== GENERAL CONFIG =====
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

quantizations=(
	"float16"
	"dynamic_range"
	"full_integer"
)

# ===== LATENCY TESTS =====
for model in "${models[@]}"
do
	for quantization in "${quantizations[@]}"
	do
		log_file_loc="./results/ptq/latencies/tflite/DEF_${model}_${quantization}.txt"
		quantized_path="./generated/ptq_tflite/DEF_PQ_${model}/run_1/PQ_${model}_${quantization}.tflite"
		benchmarker="$HOME/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"

		for i in {1..10}
		do
			$benchmarker --graph=${quantized_path} --warmup_runs=10 --num_runs=1000 >> $log_file_loc
			printf "\n-------------------------\n" >> $log_file_loc
		done
	done
done
