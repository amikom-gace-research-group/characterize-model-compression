#!/bin/bash

source ~/garuda_env/bin/activate

# ===== GENERAL CONFIG =====
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

engines=(
	"FP16"
	"INT8"
)

options=(
	--trt
	--warm-up 10
	--iters 1000
)

# ===== LATENCY TESTS =====
for model in "${models[@]}"
do
	for engine in "${engines[@]}"
	do
		log_file_loc="./results/ptq/latencies/tensorrt/${model}_${engine}.txt"
		engine_path="./generated/ptq_tensorrt/${model}/${model}_${engine}.engine"

		for i in {1..10}
		do
			polygraphy run ${engine_path} "${options[@]}" >> $log_file_loc
		done
	done
done
