# Investigation of the Scaling Law During Continual Pretraining of Commerical LLMs

This is the project repository for my undergraduate research with Glow.AI Sem 2 2024-2025 AY. 

Project Description: 
Continuously pre-train GPT-2 in python on both a randomly selected subset and a perplexity-limited subset from the C4 dataset train split. Since the allenai/c4 Hugging Face repository does not have a test split, we evaluate on the C4 dataset's validation set. 

# Two Training workflow options:

### 1. Run lightweight_fine_tuning.ipynb w/ perplexity heuristic (max 1 GPU) 
* This setup only trains model. We use both C4 train split and C4 validation split for training here (no evaluation step).
* Run each code block sequentially to train GPT-2 on C4 and observe training loss. 
1. Pip install all requirements 
2. Load the model and tokenizer 
3. Load c4/allenai dataset from Hugging Face.
4. Extract a training subset and tokenize - make sure to set the parameter ``heuristic`` to True if you choose to apply perplexity filtering to your subset selection. Feel free to modify the perplexity ranges in block 4.1. 
5. Train model with a validation set. 
6. Evaluate on downstream task. 
7. Plot perplexities to see if updates to perplexity ranges are necessary. 


### 2. Training + Evaluation on c4 validation split using .py files (recommended)
* Prerequisites: 
    * Ensure you are connected to a server with multiple GPUs to run multi-GPU distributed training by running ``nvidia-smi`` in the command line. 
    * Make sure you are in the right directory: ``cd scaling_law``
    * Run ``pip install -r requirements.txt`` to download all required packages. 

#### Steps for Perplexity/Random Subset Selection Workflow

1. Set the parameter ``heuristic=True`` for perplexity filtering and ``heuristic=False`` for random selection on line 102 in train_gpt2_perplexity.py. 

2. Run ``torchrun --nproc_per_node=<number of GPUs> train_gpt2_perplexity.py`` in the command line. 

3. The trained model will be saved to the path specified by ``output_dir`` in training_args = TrainingArguments() in each respective training file. 

#### Steps for Tracin Workflow

1. Tracin/influence  --> ``torchrun --nproc_per_node=<number of GPUs> train_gpt2_tracin.py``

2. The trained model will be saved to the path specified by ``output_dir`` in training_args = TrainingArguments() in each respective training file. 

3. We implemented a trainer called TracinTrainer and a custom callback called TracinCallback. These allow the tracin checkpoint gradients to be saved concurrently as the training loop executes. 

4. The gradients are saved to this path: "datasets/tracin_grads_1e5.pt" (feel free to modify the file name as this one assumes the training dataset is of size 1e5 tokens)

5. Run ``python -m tracin_influence_scores`` to compute influence scores from tracin gradients (Note: this takes a long time as each influence score is being computed sequentially.)

6. Navigate to datasets/influence_scores/influence_scores_list_{len(train_dataset)}.json" to see influence scores.

7. Run ``torchrun --nproc_per_node=<number of GPUs> train_with_tracin_results.py`` to extract a top k percentile subset (specified by token_max parameter in function tokenize_train) from your original subset and train a new GPT-2 model on this influence-score-filtered subset, effectively removing samples that barely influence your training. 


### Evaluating model 
1. Navigate to eval_gpt2.py and setting the variable ``checkpoint=<your target model's path>`` on line 26. 

2. Run ``python -m eval_gpt2`` to evaluate on CPU (small test dataset) or ``torchrun --nproc_per_node=<number of GPUs> eval_gpt2.py`` to evaluate model on GPU (large test dataset) --> Feel free to generate a larger test dataset by uncommenting lines 97-122 and commenting out line 123. Default dataset is 1e6 tokens large. 


## Troubleshooting

1. Make sure that in TrainingArguments: 
    1. ``per_device_train_batch_size`` is a multiple of the number of GPUs being used, otherwise the you will get the follwowing error when attempting to train on multiple GPUs: ``runtime error: chunk expects at least a 1-dimensional tensor``. Note: Jupyter notebook only supports running on 1 GPU.
    2. Adjust lr to new_lr = original_lr * (new_batch_size / original_batch_size) if the loss is not decreasing at all. 
    3. Set ``fp16=True`` to save GPU memory.

2. Adjust GPU usage if you are getting something like the following error: ``torch.OutOfMemoryError: CUDA out of memory... ``
* Try the following: 
    1. Run nvidia-smi to see the gpu memory usage
    2. In the .env file or in the below script, specify which gpus you want to train on e.g. CUDA_VISIBLE_DEVICES=0
    3. Run this bash script in the terminal:  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4   --nnodes=1 --node_rank=0   --master_addr=localhost --master_port=12345  train_gpt2.py

3. If you get this error ``torch.distributed.elastic.multiprocessing.errors.ChildFailedError or other miscellaneous errors``
* Wait a few seconds and then attempt to run the script again.  
* Reset your IDE. 

4. Make sure to update all path names in each file when dealing with different models or dataset sizes. 


