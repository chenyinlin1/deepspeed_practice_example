# deepspeed_practice
deepspeed简单入门程序

训练多个gpu并行训练：

#### 使用2个gpu并行训练：
    deepspeed --num_nodes 1 --num_gpus 2 practice.py --isDeepSpeed --deepspeed_config './deepspeed_config.json'
    
#### 只使用一个gpu训练：
    python practice.py

#### 使用zero offload训练：
    deepspeed --num_nodes 1 --num_gpus 2 practice.py --isoffload --isDeepSpeed --deepspeed_config './deepspeed_zero3_offload.json'
