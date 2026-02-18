export CUDA_VISIBLE_DEVICES=0

model_name=TINA

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/Stock/ \
  --data_path 000001.SS.csv \
  --model_id Stock_3_1 \
  --model $model_name \
  --data custom \
  --features MS \
  --target Close \
  --freq d \
  --seq_len 3 \
  --label_len 1 \
  --pred_len 1 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 256 \
  --n_heads 2 \
  --e_layers 3 \
  --factor 3 \
  --patch_len 3 \
  --na_kernel_size 9 \
  --batch_size 64 \
  --learning_rate 2.6752573061452198e-05 \
  --dropout 0.030875537254563414 \
  --patience 2 \
  --train_epochs 3 \
  --lradj type2 \
  --activation swiglu \
  --inverse \
  --des 'Exp' \
  --itr 1
