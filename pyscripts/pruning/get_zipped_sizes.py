# BIARKAN DULU!

import tempfile
import os
import argparse
import zipfile
import shutil
import pathlib

def get_gzipped_tflite_size(file):
    # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(file)
    
    return os.path.getsize(zipped_file)


def get_zipped_model_size(model_folder):
    _, zipped_file = tempfile.mkstemp()
    shutil.make_archive(zipped_file, 'zip', model_folder)

    return os.path.getsize(f'{zipped_file}.zip')


def main(args):
    base_dir=f'{pathlib.Path.cwd()}/generated'

    saved_model_loc=f'{base_dir}/baselines/{args.model_name}/run_{args.run_num}'
    pruned_keras_model_loc=f'{base_dir}/pruned_keras/{args.model_name}_{args.sparsity}_sparsity/run_{args.run_num}'

    print("\nSize of gzipped baseline Keras model: %.2f bytes" % (get_zipped_model_size(saved_model_loc)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_zipped_model_size(pruned_keras_model_loc)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name')
    parser.add_argument('--run_num')
    parser.add_argument('--sparsity')

    args = parser.parse_args()

    main(args)
