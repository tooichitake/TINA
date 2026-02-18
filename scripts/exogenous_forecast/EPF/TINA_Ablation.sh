export CUDA_VISIBLE_DEVICES=0

python -u ablation_epf_tina.py \
  --datasets all \
  --variants all \
  --results_file results_ablation_epf.txt
