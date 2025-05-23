## Преобразование прозы в стихи

pip install transformers==4.38.2 accelerate bitsandbytes peft==0.9.0 wandb numpy pandas pytorch datasets trl

tmux attach -t distler

python3 train/train.py --pretrain --model='qwen' --save_steps=5000 --train_dataset=dataset/trainset_pretrain --epochs=2 --log_steps=200 --markup=rhyme_markup --warmup_steps=320 --lr=2e-5

python3 train/train.py --model='qwen' --save_steps=2000 --from_pretrain=output/qwen-05-18-09-32-pretrain/checkpoint-5369 --lr=1e-6 --epochs=10 --log_steps=200

python3 train/train.py --model='qwen' --save_steps=2000 --from_pretrain=output/qwen-05-22-17-10-pretrain/checkpoint-10738 --epochs=10 --log_steps=200 --markup=rhyme_markup_long --warmup_steps=320 --lr=1e-5

python3 train/train.py --model='t-lite' --save_steps=2000 --from_pretrain=output/t-lite-05-18-09-39-pretrain/checkpoint-5369 --lr=1e-6 --epochs=10 --log_steps=200
