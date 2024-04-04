#!/bin/bash

source ~/garuda_env/bin/activate

# DenseNet
module_name="mobilenet_v3"
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

for model in "${models[@]}"
do
	onnx_file_location="./generated/onnx/${model}.onnx"
	log_output="./results/ptq/tensorrt/${model}.txt"
	for i in {1..5} 
	do
		tfonnx_options=(
			# module name / preprocessing options
			--saved-model ./generated/baselines/${model}/run_${i}
			# model name
			--output ${onnx_file_location}
		)

		fp16_converter_options=(
			--fp16
			-o ./generated/ptq_tensorrt/${model}/${model}_FP16.engine
		)

		int8_convert_options=(
			# set INT8 as output
			--int8
			# set data-loader script for calibration
			--data-loader-script ./pyscripts/tensorrt_utils/mobilenet_loader.py
			# set tactic sources to match Nano's
			# --tactic-sources cublas cudnn
			# set output location
			-o ./generated/ptq_tensorrt/${model}/${model}_INT8.engine
		)

		inference_options=(
			--module_name ${module_name}
			--model_name ${model}
		)

		# generate ONNX model
		start_time=`date +%s.%N`
		python3 -m tf2onnx.convert "${tfonnx_options[@]}"
		end_time=`date +%s.%N`
		runtime=$(echo "$end_time - $start_time" | bc -l)
		printf "\nONNX conversion took %s seconds!" "$runtime" >> $log_output

		# run polygraphy to get TRT engine
		start_time=`date +%s.%N`
		polygraphy convert ${onnx_file_location} "${fp16_converter_options[@]}"
		end_time=`date +%s.%N`
		runtime=$(echo "$end_time - $start_time" | bc -l)
		printf "\nFP16 TRT engine done in %s seconds!" "$runtime" >> $log_output

		start_time=`date +%s.%N`
		polygraphy convert ${onnx_file_location} "${int8_convert_options[@]}"
		end_time=`date +%s.%N`
		runtime=$(echo "$end_time - $start_time" | bc -l)
		printf "\nINT8 TRT engine done in %s seconds!\n" "$runtime" >> $log_output

		# Inference
		python3 pyscripts/tensorrt_utils/ptq_engine_infer.py "${inference_options[@]}" >> $log_output
		printf "\n----------------------------\n" >> $log_output
	done
done
