set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python train/unimodalvla_phase1_train.py \
  --output_dir outputs/unimodalvla_phase1_dummy \
  --overwrite_output_dir True \
  --do_train True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --save_steps 10 \
  --bf16 False \
  --remove_unused_columns False
