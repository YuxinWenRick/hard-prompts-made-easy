from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import torch.nn as nn
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os, copy
from datasets import load_dataset
from openprompt.plms import load_plm
from utils import nn_project
import wandb
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor  # use Adafactor is the default setting for T5

########### 
# Adapted from https://github.com/thunlp/OpenPrompt/blob/main/tutorial/1.4_soft_template.py 
###########

### Note some boolean arguments are string or ints to comply with submittit input formating
### https://github.com/facebookincubator/submitit

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", type=bool, default=True, help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument("--model", type=str, default='gpt2')
parser.add_argument("--model_name_or_path", default='gpt2-large')
parser.add_argument("--project_root", default="", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int)
parser.add_argument("--verbalizer_id", type=int)
parser.add_argument("--data_dir", type=str, default="") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions.
parser.add_argument("--dataset",type=str)
parser.add_argument("--dataset_holdout",type=int, default=5000)
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--init_from_vocab", type=bool, default=False)
parser.add_argument("--eval_every_steps", type=int, default=500)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--soft_token_num", type=int, default=100)
parser.add_argument("--optimizer", type=str, default="Adafactor")
parser.add_argument("--PEZ_discrete_prompt", type=int, default=0, help="(int 1/0) 1 if using PEZ algorithm to create soft prompt, else soft prompt training; default is 0")
parser.add_argument("--fluentPrompt", type=int, default=0, help="(int 1/0) 1 if using fluentPrompt algorithm to create soft prompt,; default is 0")
parser.add_argument("--fluency_weight", type=float, default=0, help="(int 1/0) 1 if using fluency constraint to create discrete prompt,; default is 0")
parser.add_argument("--beta_zero", type=int, default=0, help="(int 1/0) 1 if noise term turned off; default is 0")
parser.add_argument("--soft_prompt", type=float, default=0, help="(int 1/0) 1 if use soft prompt for evaluation; default is 0")
parser.add_argument("--with_tracking", type=str, default="False")
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()
print(args)

content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"model_name_or_path {args.model_name_or_path}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += f"PEZ_discrete_prompt {bool(args.PEZ_discrete_prompt)}\t"
content_write += "\n"

print(content_write)

import random
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)
print(args)
if args.with_tracking == "True":
    wandb.init(project="Hard_Prompts_Made_Easy", config=args)
    if args.run_name is not None:
        wandb.run.name = args.run_name

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
dataset = {}
print(tokenizer)
# Below are multiple dataset examples, including few-shot ones.
if args.dataset == "sst2":
    raw_dataset = load_dataset("glue", "sst2")
    dataset = {}
    train_dataset_size = len(raw_dataset['train'])
    holdout_indicies = np.random.choice(train_dataset_size, args.dataset_holdout, replace=False)
    dataset["holdout"] = []
    for split in ['train', 'validation', 'test']:
        dataset[split] = []
        for idx,data in enumerate(raw_dataset[split]):

            input_example = InputExample(text_a = data['sentence'], label=int(data['label']), guid=idx)
            if split == 'train' and idx in holdout_indicies:
                dataset["holdout"].append(input_example)
            else:
                dataset[split].append(input_example)
    class_labels = ["negative", "positive"]
    scriptsbase = "temp_and_verb/SST2"
    scriptformat = "txt"
    max_seq_l = 128 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.train_batch_size
        batchsize_e = args.eval_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        model_parallelize = True # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.train_batch_size
        batchsize_e = args.eval_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        model_parallelize = False
elif args.dataset == "ag_news":
    raw_dataset = load_dataset("ag_news")
    dataset = {}
    dataset["holdout"] = []
    dataset["validation"] = []
    
    train_dataset_size = len(raw_dataset['train'])
    holdout_indicies = np.random.choice(train_dataset_size, args.dataset_holdout, replace=False)
    for split in ['train', 'test']:
        dataset[split] = []
        for idx,data in enumerate(raw_dataset[split]):
            input_example = InputExample(text_a = data['text'], label=int(data['label']), guid=idx)     
            # To account for no validation set; turn test into validation so functions with the rest of the code
            if split == 'test':
                dataset["validation"].append(input_example) 
            elif split == 'train' and idx in holdout_indicies:
                dataset["holdout"].append(input_example)
            else:
                dataset[split].append(input_example)

    class_labels = ["politics", "sports", "business", "technology"]
    scriptsbase = "temp_and_verb/agnews"
    scriptformat = "txt"
    max_seq_l = 256 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.train_batch_size
        batchsize_e = args.eval_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        model_parallelize = True # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.train_batch_size
        batchsize_e = args.eval_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        model_parallelize = False
elif args.dataset == "amazon":
    raw_dataset = load_dataset("amazon_polarity")
    dataset = {}
    dataset["holdout"] = []
    dataset["validation"] = []
    test_dataset_size = len(raw_dataset['test'])
    holdout_indicies = np.random.choice(len(raw_dataset['train']), args.dataset_holdout, replace=False)
    train_indicies = np.random.choice(len(raw_dataset['train']), 2*args.dataset_holdout+32*(args.max_steps+1)) # only tokenize however many we need 
    for split in ['train', 'test']:
        dataset[split] = []
        for idx,data in enumerate(raw_dataset[split]):
            if split == 'train' and idx not in [train_indicies, holdout_indicies]: continue
            input_example = InputExample(text_a = data['title']+" "+data['content'], label=int(data['label']), guid=idx)
            # To account for no validation set; turn test into validation so functions with the rest of the code
            if split == 'train' and idx in holdout_indicies: 
                dataset["holdout"].append(input_example) 
            elif split == 'train' and idx in train_indicies:
                dataset["train"].append(input_example)
            elif split == 'test':
                dataset["validation"].append(input_example)

    class_labels = ["negative", "positive"]
    scriptsbase = "temp_and_verb/amazon"
    scriptformat = "txt"
    max_seq_l = 256 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.train_batch_size
        batchsize_e = args.eval_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        model_parallelize = True # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.train_batch_size
        batchsize_e = args.eval_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        model_parallelize = False
else:
    raise NotImplementedError


# Now define the template and verbalizer.
# Note that soft template can be combined with hard template, by loading the hard template from file.
# For example, the template in soft_template.txt is {}
# The choice_id 1 is the hard template
mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab, initialize_from_classes=True, classes=class_labels).from_file(f"{scriptsbase}/soft_template.txt", choice=args.template_id)
# Sometimes that template does not load properly. This is in case it does not. This is an issue with OpenPrompt
if args.dataset == 'sst2':  
    mytemplate.text[1]['text'] = ". It was "
    mytemplate.text[1]['add_prefix_space'] = ""
elif args.dataset == 'ag_news':
    mytemplate.text[1]['text'] = "It was about "
    mytemplate.text[1]['add_prefix_space'] = " "
elif args.dataset == 'amazon':
    mytemplate.text[1]['text'] = "It was "
    mytemplate.text[1]['add_prefix_space'] = " "
elif args.dataset == 'yahoo':
    mytemplate.text[1]['text'] = "It was about "
    mytemplate.text[1]['add_prefix_space'] = " "

myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)
wrapped_example = mytemplate.wrap_one_example(dataset['train'][1])
print(wrapped_example)

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

if model_parallelize:
    prompt_model.parallelize()

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=5,
    batch_size=batchsize_t,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=5,
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
holdout_dataloader = PromptDataLoader(dataset=dataset["holdout"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=5,
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

print("Train truncate rate: {}".format(train_dataloader.tokenizer_wrapper.truncate_rate), flush=True)
print("Val truncate rate: {}".format(validation_dataloader.tokenizer_wrapper.truncate_rate), flush=True)
print("Holdout truncate rate: {}".format(holdout_dataloader.tokenizer_wrapper.truncate_rate), flush=True)

def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    total_loss_val = 0
    log_perplexity = 0
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            if step == 0 and 'gpt' in args.model: 
                batch = copy.deepcopy(inputs)   
                batch = prompt_model.template.process_batch(batch)
                input_batch = {key: batch[key] for key in batch if key in prompt_model.prompt_model.forward_keys}
                outputs = prompt_model.plm(**input_batch, output_hidden_states=True)
                prompt_logits = outputs['logits'][:,:args.soft_token_num,:].clone()
                ppl_labels = nn_indices.repeat(len(batch['label']),1)
                # used https://github.com/huggingface/transformers/blob/5db9abde439bc02c3791da2a4fefee80d94d5b96/src/transformers/models/gpt2/modeling_gpt2.py#L1073 for perplexity
                shift_logits = prompt_logits[..., :-1, :].contiguous()
                shift_labels = ppl_labels[..., 1:].contiguous()
                log_perplexity = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            total_loss_val += loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    print("Prediction: ", allpreds[:10])
    print("Labels: ", alllabels[:10])
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc, total_loss_val/step, log_perplexity

loss_func = torch.nn.CrossEntropyLoss()
tot_step = args.max_steps


if args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=500, num_training_steps=tot_step)
else:
    optimizer1 = None
    scheduler1 = None


optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
if args.optimizer.lower() == "adafactor":
    optimizer2 = Adafactor(optimizer_grouped_parameters2,
                            lr=args.prompt_lr,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
elif args.optimizer.lower() == "adamw":
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr) # usually lr = 0.5
    scheduler2 = get_linear_schedule_with_warmup(
                    optimizer2,
                    num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500


tot_loss = 0
log_loss = 0
best_val_acc = 0
best_holdout_acc = 0
best_val_acc_via_holdout = 0
glb_step = 0
actual_step = 0
val_acc = -1
val_loss = -1 # -1 not np.inf so it plots in wandb
leave_training = False
best_prompt_str = ""
best_prompt_tokens = []
best_prompt_str_holdout = ""
best_prompt_tokens_holdout = []
best_log_PPL_holdout = -1
acc_traces = []
fluency_loss = 0
label_loss = 0
if bool(args.fluentPrompt): 
    beta_decay_rate = np.exp((np.log(0.0001)/args.max_steps)) # beta starts at 1 and decays to 0.0001 using geometric degradation
    print("Beta Decay Rate: ", beta_decay_rate)

if args.with_tracking == "True": prompt_table = wandb.Table(columns=["prompt","tokens", "holdout_acc", "log PPL", "holdout_loss", "step"])

tot_train_time = 0
pbar_update_freq = 10
prompt_model.train()
pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(1000000):
    print(f"Begin epoch {epoch}")
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()

        if glb_step % 10 == 0 and actual_step % gradient_accumulation_steps == 0: 
            projected_embeds, nn_indices = nn_project(prompt_model.template.soft_embeds, prompt_model.template.raw_embedding)
            print("Front part prompt: ", tokenizer.decode(nn_indices))
            if args.fluency_weight != 0:
                print("Fluency Loss: ", fluency_loss)
                print("Label Loss: ", label_loss)

        if bool(args.PEZ_discrete_prompt) and actual_step % gradient_accumulation_steps == 0:
            ### COPY SOFT PROMPT ###
            soft_prompt_copy = prompt_model.template.soft_embeds.clone().detach()
            ### PROJECT ###
            projected_embeds, nn_indices = nn_project(prompt_model.template.soft_embeds, prompt_model.template.raw_embedding)
            ### PROJECTED EMBEDDING FOR FORWARD ###
            prompt_model.template.soft_embeds.data = projected_embeds.data

        tot_train_time -= time.time()

        if args.fluency_weight != 0:
            # taken from the source OpenPrompt code
            # prepare batch
            batch = inputs
            batch = prompt_model.template.process_batch(batch)
            input_batch = {key: batch[key] for key in batch if key in prompt_model.prompt_model.forward_keys}
            # get outputs from batch
            outputs = prompt_model.plm(**input_batch, output_hidden_states=True)
            # get prompt logits
            prompt_logits = outputs['logits'][:,:args.soft_token_num,:].clone()
            ppl_labels = nn_indices.repeat(len(batch['label']),1)
            # used for GPT2 https://github.com/huggingface/transformers/blob/5db9abde439bc02c3791da2a4fefee80d94d5b96/src/transformers/models/gpt2/modeling_gpt2.py#L1073 for perplexity
            shift_logits = prompt_logits[..., :-1, :].contiguous()
            shift_labels = ppl_labels[..., 1:].contiguous()
            fluency_loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # from Open Prompt code again
            outputs = prompt_model.template.post_processing_outputs(outputs)
            outputs = prompt_model.verbalizer.gather_outputs(outputs)
            if isinstance(outputs, tuple):
                outputs_at_mask = [prompt_model.extract_at_mask(output, batch) for output in outputs]
            else:
                outputs_at_mask = prompt_model.extract_at_mask(outputs, batch)
            label_words_logits = prompt_model.verbalizer.process_outputs(outputs_at_mask, batch=batch)
            label_loss = loss_func(label_words_logits, batch['label'])
            fluency_weight = args.fluency_weight
            loss = fluency_weight*fluency_loss + (1-fluency_weight)*label_loss
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1
        else:
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1

        ### REWRITE SOFT PROMPT FOR UPDATE ###
        if bool(args.PEZ_discrete_prompt) and actual_step % gradient_accumulation_steps == 0 and actual_step > gradient_accumulation_steps:
            prompt_model.template.soft_embeds.data = soft_prompt_copy.data
        
        if not bool(args.fluentPrompt) and actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

            if optimizer1 is not None:
                optimizer1.step()
                optimizer1.zero_grad()
            if scheduler1 is not None:
                scheduler1.step()
            if optimizer2 is not None:
                optimizer2.step()
                optimizer2.zero_grad()
            if scheduler2 is not None:
                scheduler2.step()

        elif bool(args.fluentPrompt) and actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

            # ei = ProjE[ ̃ei−1 − η∇ ̃eE( ̃ei−1) + √(2ηβ)z]
            # we will calculate noise term add to the grad descent step (ei−1 − η∇eE( ̃ei−1))
            # calculate noise term
            learning_rate_i = scheduler2.get_last_lr()[0]
            beta_i = 1*(beta_decay_rate**(glb_step-1)) # note beta is 1 at step 1 and 0.0001 at 5001 if max steps is 5000; cause glb_step is incremented first
            # follwoing line from: https://github.com/Sachin19/mucoco/blob/05586078e6a916ebfba7d7643891e95bc0d5dc72/mucoco/utils/optim.py#L1194
            z = torch.normal(
                        mean=torch.zeros_like(prompt_model.template.soft_embeds.data),
                        std=torch.ones_like(prompt_model.template.soft_embeds.data)
                    )

            ###### TURN OFF NOISE TERM if beta_zero is 1 ######
            if bool(args.beta_zero): beta_i = 0 
            noise_term = ((2 * learning_rate_i * beta_i) ** 0.5) * z
            # grad descent step
            optimizer2.step()
            optimizer2.zero_grad()
            scheduler2.step()
            # add noise before project
            prompt_model.template.soft_embeds.data = prompt_model.template.soft_embeds.data+noise_term.cuda()
            # project
            projected_embeds, nn_indices = nn_project(prompt_model.template.soft_embeds, prompt_model.template.raw_embedding)
            # update soft_embed to projected embedding
            prompt_model.template.soft_embeds.data = projected_embeds.data

        tot_train_time += time.time()

        if actual_step % gradient_accumulation_steps == 0 and glb_step > 0 and glb_step % args.eval_every_steps == 0: 
            if bool(args.PEZ_discrete_prompt):
                # get new projected prompt for evaluation on holdout set
                soft_prompt_copy = prompt_model.template.soft_embeds.clone().detach()
                projected_embeds, nn_indices = nn_project(prompt_model.template.soft_embeds, prompt_model.template.raw_embedding)
                prompt_model.template.soft_embeds.data = projected_embeds.data
                text_prompt = tokenizer.decode(nn_indices)
                print("Evaluating Front part prompt: ", text_prompt)


            holdout_acc, holdout_loss, log_perplexity_eval = evaluate(prompt_model, holdout_dataloader, desc="Hold Out")
            projected_embeds, nn_indices = nn_project(prompt_model.template.soft_embeds, prompt_model.template.raw_embedding)
            text_prompt = tokenizer.decode(nn_indices)
            
            print("Holdout Acc: ", holdout_acc)
            print("Holdout Loss: ", holdout_loss)
            print("Log Perpleixty: ", log_perplexity_eval)
            ### REWRITE SOFT PROMPT FOR UPDATE ###
            if bool(args.PEZ_discrete_prompt) and not leave_training:
                prompt_model.template.soft_embeds.data = soft_prompt_copy.data
            
            if holdout_acc >= best_holdout_acc:
                best_val_acc_via_holdout = val_acc
                best_holdout_acc = holdout_acc
                best_prompt_str_holdout = tokenizer.decode(nn_indices)
                best_prompt_tokens_holdout = nn_indices
                best_log_PPL_holdout = log_perplexity_eval
                best_projected_embedding = projected_embeds.data.clone()

            acc_traces.append(val_acc)
            print("Glb_step {}, val_acc {}, holdout_acc {}, average time {}".format(glb_step, val_acc, holdout_acc, tot_train_time/actual_step), flush=True)
            if args.with_tracking == "True": 
                wandb.log({"global_steps": glb_step, "epoch": epoch, "train_loss": tot_loss/glb_step, "val_acc": val_acc, "holdout_acc": holdout_acc, "val_loss": val_loss, "holdout_loss": holdout_loss, "log PPL": log_perplexity_eval}, step=glb_step)
                prompt_table.add_data(text_prompt, nn_indices, holdout_acc, log_perplexity_eval, holdout_loss, glb_step) 
            
            prompt_model.train()
        
        if bool(args.PEZ_discrete_prompt) and actual_step % gradient_accumulation_steps == 0:
            ### COPY SOFT PROMPT ###
            soft_prompt_copy = prompt_model.template.soft_embeds.clone().detach()
            ### PROJECT ###
            projected_embeds, nn_indices = nn_project(prompt_model.template.soft_embeds, prompt_model.template.raw_embedding)
            ### PROJECTED EMBEDDING FOR FORWARD ###
            prompt_model.template.soft_embeds.data = projected_embeds.data
        
        if glb_step > args.max_steps:
            leave_training = True
            break

    if leave_training:
        break
    
print(f"========== Running Eval for {args.dataset} Dataset ==========")
if  not bool(args.soft_prompt): prompt_model.template.soft_embeds.data = best_projected_embedding
holdout_acc, holdout_loss, holdout_log_perplexity = evaluate(prompt_model, holdout_dataloader, desc="Hold Out")
print(f"Sanity Best Hold out acc {best_holdout_acc} with this holdout acc {holdout_acc}")
print(f"Are they the same? {best_holdout_acc == holdout_acc}")
val_acc, val_loss, val_log_perplexity = evaluate(prompt_model, validation_dataloader, desc="Valid")
print("Val Acc: ", val_acc)
print("Val Loss: ", val_loss)
print("Pre Update Log Perplexity: ", fluency_loss)
best_val_acc_via_holdout = val_acc
best_val_acc = -1

if args.with_tracking == "True":
    wandb.log({"training_samples" : prompt_table})


content_write += f"BestValAcc:{best_val_acc}\tBestValAcc(From Holdout):{best_val_acc_via_holdout}\tEndValAcc:{acc_traces[-1]}\tLog PPL:{best_log_PPL_holdout}\n"
content_write += "\n"
if args.with_tracking == "True": wandb.log({"Best Val Acc": best_val_acc, "Best Val Acc (From Holdout)": best_val_acc_via_holdout, "Best Log PPL (From Holdout)": best_log_PPL_holdout})
print(content_write)