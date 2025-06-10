## Преобразование прозы в стихи

pip install -r requirements.txt

apt-get install espeak -y

tmux attach -t distler

python3 prose-to-poetry/train.py --pretrain --model='qwen' --save_steps=5000 --train_dataset=dataset/trainset_pretrain --epochs=2 --log_steps=200 --markup=rhyme_markup --warmup_steps=320 --lr=2e-5

python3 prose-to-poetry/train.py --model='qwen' --save_steps=2000 --from_pretrain=output/qwen-05-18-09-32-pretrain/checkpoint-5369 --lr=1e-6 --epochs=10 --log_steps=200

python3 prose-to-poetry/train.py --model='qwen' --save_steps=2000 --from_pretrain=output/qwen-05-22-17-18-pretrain/checkpoint-10738 --epochs=5 --log_steps=200 --markup=rhyme_markup --warmup_steps=30 --lr=1e-6

23 - stanzas

python3 prose-to-poetry/train.py --model='t-lite' --save_steps=2000 --from_pretrain=output/t-lite-05-18-09-39-pretrain/checkpoint-5369 --lr=1e-6 --epochs=10 --log_steps=200

python3 prose-to-poetry/eval.py --name=qwen --model=qwen --checkpoint=output/qwen-05-23-22-32/checkpoint-624 --markup=rhyme_markup

python3 prose-to-poetry/eval.py --name=qwen_generate --model=qwen --checkpoint=output/qwen-05-22-17-18-pretrain/checkpoint-10738 --markup=rhyme_markup --generate
