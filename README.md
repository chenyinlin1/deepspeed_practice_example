# deepspeed_practice
deepspeed简单入门程序

训练多个gpu并行训练：
deepspeed --num_nodes=1 --num_gpus 2 train_ddpg.py --isDeepSpeed --deepspeed_config './deepspeed_config.json'
