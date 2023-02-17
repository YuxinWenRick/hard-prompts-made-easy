# To generate the prompts from a full dataset running the following lines

Note that PEZ_discrete_prompt refers to our method while fluentPrompt refers to FluentPrompt and when fluentPrompt and zero_beta is set to 1 refers to the AutoPrompt SGD version. To log runs using wandb, add "--with_tracking True" flag


To run our baseline, you can run the following line
```
conda create --name PEZ python=3.9
pip install -r requirements.txt
python setup.py install

export datasetname=sst2
export PEZ_discrete_prompt=1

python src/train_prompts.py --model=gpt2 --plm_eval_mode=True --template_id=0 --verbalizer_id=0 --warmup_step_prompt=0 --model_name_or_path=gpt2-large --dataset=$datasetname --dataset_holdout=5000 --train_batch_size 16 --eval_batch_size 16 --gradient_accumulation_steps 1 --eval_every_steps=100 --fluency_weight=0 --max_steps=5000 --optimizer=Adafactor --prompt_lr=0.3 --seed=100 --soft_token_num=10 --PEZ_discrete_prompt=$PEZ_discrete_prompt
```