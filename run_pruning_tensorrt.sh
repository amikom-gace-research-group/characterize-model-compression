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
		onnx_file_location="./generated/onnx/pruned_${model}.onnx"
		log_output="./results/pruning/gpu_tensorrt/${model}_${sparsity}_sparsity.txt"

		for i in {1..5} 
		do
			pruning_options=(
				--module_name ${module_name}
				--model_name ${model}
				--target_sparsity ${sparsity}
				--run_num "${i}"
				--tuner GPU
			)

			tfonnx_options=(
				--saved-model ./generated/pruned_keras/${model}_${sparsity}_sparsity/run_${i}
				--output ${onnx_file_location}
			)

			int8_convert_options=(
				--int8
				--sparse-weights
				--data-loader-script ./pyscripts/tensorrt_utils/data_loader.py
				-o ./generated/pruned_tensorrt/${model}_${sparsity}_sparsity.engine
			)

			inference_options=(
				--sparsity ${sparsity}
				--module_name ${module_name}
				--model_name ${model}
			)

			# Run pruning in python
			python3 pyscripts/pruning/mobilenet_prune.py "${pruning_options[@]}" >> $log_output

			# generate ONNX model
			start_time=`date +%s.%N`
			python3 -m tf2onnx.convert "${tfonnx_options[@]}"
			end_time=`date +%s.%N`
			runtime=$(echo "$end_time - $start_time" | bc -l)
			printf "\nONNX conversion took %s seconds!" "$runtime" >> $log_output

			start_time=`date +%s.%N`
			polygraphy convert ${onnx_file_location} "${int8_convert_options[@]}"
			end_time=`date +%s.%N`
			runtime=$(echo "$end_time - $start_time" | bc -l)
			printf "\nINT8 TRT engine done in %s seconds!\n" "$runtime" >> $log_output

			# Inference
			python3 pyscripts/tensorrt_utils/pruned_engine_infer.py "${inference_options[@]}" >> $log_output
			printf "\n----------------------------\n" >> $log_output
		done
	done
done

module_name="densenet"
models=(
	"DenseNet121"
	"DenseNet169"
	"DenseNet201"
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
		onnx_file_location="./generated/onnx/pruned_${model}.onnx"
		log_output="./results/pruning/gpu_tensorrt/${model}_${sparsity}_sparsity.txt"

		for i in {1..5} 
		do
			pruning_options=(
				--module_name ${module_name}
				--model_name ${model}
				--target_sparsity ${sparsity}
				--run_num "${i}"
				--tuner GPU
			)

			tfonnx_options=(
				--saved-model ./generated/pruned_keras/${model}_${sparsity}_sparsity/run_${i}
				--output ${onnx_file_location}
			)

			int8_convert_options=(
				--int8
				--sparse-weights
				--data-loader-script ./pyscripts/tensorrt_utils/data_loader.py
				-o ./generated/pruned_tensorrt/${model}_${sparsity}_sparsity.engine
			)

			inference_options=(
				--sparsity ${sparsity}
				--module_name ${module_name}
				--model_name ${model}
			)

			# Run pruning in python
			python3 pyscripts/pruning/prune.py "${pruning_options[@]}" >> $log_output

			# generate ONNX model
			start_time=`date +%s.%N`
			python3 -m tf2onnx.convert "${tfonnx_options[@]}"
			end_time=`date +%s.%N`
			runtime=$(echo "$end_time - $start_time" | bc -l)
			printf "\nONNX conversion took %s seconds!" "$runtime" >> $log_output

			start_time=`date +%s.%N`
			polygraphy convert ${onnx_file_location} "${int8_convert_options[@]}"
			end_time=`date +%s.%N`
			runtime=$(echo "$end_time - $start_time" | bc -l)
			printf "\nINT8 TRT engine done in %s seconds!\n" "$runtime" >> $log_output

			# Inference
			python3 pyscripts/tensorrt_utils/pruned_engine_infer.py "${inference_options[@]}" >> $log_output
			printf "\n----------------------------\n" >> $log_output
		done
	done
done

module_name="efficientnet"
models=(
	"EfficientNetB0"
	"EfficientNetB1"
	"EfficientNetB2"
	"EfficientNetB3"
	"EfficientNetB4"
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
		onnx_file_location="./generated/onnx/pruned_${model}.onnx"
		log_output="./results/pruning/gpu_tensorrt/${model}_${sparsity}_sparsity.txt"

		for i in {1..5} 
		do
			pruning_options=(
				--module_name ${module_name}
				--model_name ${model}
				--target_sparsity ${sparsity}
				--run_num "${i}"
				--tuner GPU
			)

			tfonnx_options=(
				--saved-model ./generated/pruned_keras/${model}_${sparsity}_sparsity/run_${i}
				--output ${onnx_file_location}
			)

			int8_convert_options=(
				--int8
				--sparse-weights
				--data-loader-script ./pyscripts/tensorrt_utils/data_loader.py
				-o ./generated/pruned_tensorrt/${model}_${sparsity}_sparsity.engine
			)

			inference_options=(
				--sparsity ${sparsity}
				--module_name ${module_name}
				--model_name ${model}
			)

			# Run pruning in python
			python3 pyscripts/pruning/prune.py "${pruning_options[@]}" >> $log_output

			# generate ONNX model
			start_time=`date +%s.%N`
			python3 -m tf2onnx.convert "${tfonnx_options[@]}"
			end_time=`date +%s.%N`
			runtime=$(echo "$end_time - $start_time" | bc -l)
			printf "\nONNX conversion took %s seconds!" "$runtime" >> $log_output

			start_time=`date +%s.%N`
			polygraphy convert ${onnx_file_location} "${int8_convert_options[@]}"
			end_time=`date +%s.%N`
			runtime=$(echo "$end_time - $start_time" | bc -l)
			printf "\nINT8 TRT engine done in %s seconds!\n" "$runtime" >> $log_output

			# Inference
			python3 pyscripts/tensorrt_utils/pruned_engine_infer.py "${inference_options[@]}" >> $log_output
			printf "\n----------------------------\n" >> $log_output
		done
	done
done
