#!/bin/bash

source ~/garuda_env/bin/activate

# ===== GENERAL CONFIG =====
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
	# "DenseNet121"
	# "DenseNet169"
	# "DenseNet201"
	# "EfficientNetB0"
	# "EfficientNetB1"
	# "EfficientNetB2"
	# "EfficientNetB3"
	# "EfficientNetB4"
)

sparsities=(
	"0.25"
	"0.50"
	"0.75"
)

options=(
	--trt
	--warm-up 10
	--iters 1000
)

# ===== LATENCY TESTS =====
for model in "${models[@]}"
do
	for sparsity in "${sparsities[@]}"
	do
		log_file_loc="./results/pruning/latencies/tensorrt/${model}_${sparsity}.txt"
		engine_path="./generated/pruned_tensorrt/${model}_${sparsity}_sparsity.engine"

		for i in {1..10}
		do
			polygraphy run ${engine_path} "${options[@]}" >> $log_file_loc
		done
	done
done
