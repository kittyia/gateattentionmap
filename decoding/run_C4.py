from settings_manager import *

import os
import json

import pandas as pd
import numpy as np

from argparse import Namespace
from datetime import datetime
from pprint import pprint
from functools import partial
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import entropy

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from policy_manager import csPolicy
from accelerate import dispatch_model, infer_auto_device_map
from peft import get_peft_config, get_peft_model 
from utility import bias_measure


def load_model(args):
    """Load and return the model and tokenizer"""
    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom","llama"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer, device


def load_model_cache(args, hf_cache_dir):
    """Load and return the model and tokenizer"""
    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom","llama"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto', cache_dir=hf_cache_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', cache_dir=hf_cache_dir)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=hf_cache_dir)
    return model, tokenizer, device


def generate(idx, prompt, args, debiasing_policy=None, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    # print(f'generate with debiasing policy {debiasing_policy}')
    
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens,
                                                    tokenizer=tokenizer,
                                                    policy=debiasing_policy,
                                                    )
    watermark_processor.init_para()

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k = 0,
            top_p = 1,
            temperature=args.sampling_temp,
            num_beams=args.n_beams,
            min_length=args.max_new_tokens
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens
    
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, 
                           truncation=True, max_length=args.prompt_max_length)
    tokd_input.to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    full_output_ids_without_watermark = generate_without_watermark(**tokd_input)
    full_output_ids_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = full_output_ids_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = full_output_ids_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    output_original_scores, output_modified_scores = watermark_processor.get_output_scores()
    #print(torch.all(output_original_scores[0] == output_modified_scores[0]))

    output_stats_with_watermark = []
    assert len(output_original_scores) == len(output_modified_scores) == output_with_watermark.shape[-1]
    for i in range(len(output_original_scores)):
        next_token_id = output_with_watermark[0][i].unsqueeze(-1).unsqueeze(-1)
        next_token_logits = output_original_scores[i]
        next_token_dist = torch.softmax(next_token_logits, dim=-1)
        next_token_entropy = entropy(next_token_dist.cpu().detach().numpy().tolist()[0])
        sorted_indices = torch.argsort(next_token_dist, descending=True)
        next_token_rank = (sorted_indices == next_token_id).nonzero(as_tuple=True)[1].item() + 1
        next_token_prob = next_token_dist.squeeze()[next_token_id].item()
        output_stats_with_watermark.append({'next_token_id': next_token_id.item(), 'next_token_rank': next_token_rank, 
                                            'next_token_prob': next_token_prob, 'next_token_entropy': next_token_entropy})
    # insert
    watermark_processor.save_state_log(os.path.join(args.result_local_path, f'{idx}_{debiasing_policy.name}.csv'))

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            full_output_ids_without_watermark.detach().tolist(),
            full_output_ids_with_watermark.detach().tolist(),
            output_stats_with_watermark,
            args) 


def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
    else:
        score_dict = None
        print('Error')
    return score_dict['z_score'], score_dict['p_value'] 

    
def main(args, policy): 
    # policy = csPolicy()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    # hf_cache_dir = '/data/mjmao/ood/hf_models'
    hf_cache_dir = args.hf_cache_dir
    hf_cache_dir = '/Users/mgzhao/Documents/debias/hf_model'
    print(args)
    model, tokenizer, device = load_model_cache(args, hf_cache_dir)
    #model, tokenizer, device = load_model(args)

    prompts_file = args.prompts_file
    with open(prompts_file, "r") as input_file:
        data = json.load(input_file)

    # result_content_dir = os.path.join(args.result_global_path, {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")})
    with open(os.path.join(args.result_global_path, 'config.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    
    output_file = open(os.path.join(args.result_global_path, 'output_content.jsonl'), "w")

    i = 0
    for input_text in tqdm(data):
        
        if i > 1000:
            continue

        print(f'MarkerForANewInstance: {i}')
        input_text = (input_text)
        args.default_prompt = input_text

        # Input text/prompt:
        # print("#"*term_width)
        # print("Prompt: ", input_text)

        # Benchmark of green token list
        _, _, output_wo_green, output_green, full_output_ids_wo_green, full_output_ids_green, output_stats_green, _ = generate(idx=i,prompt=input_text,args=args, debiasing_policy=policy, model=model, device=device, tokenizer=tokenizer)
        i += 1
        # print("-"*term_width)
        # print("Output without green token list:")
        # print(output_wo_green)
        
        # print("-"*term_width)
        # print("Output with green token list:")
        # print(output_green)
        
        # print("-"*term_width)
        # print('Prompt bias', bias_measure(input_text))
        # print('Before debias', bias_measure(output_wo_green))
        # print('After debias', bias_measure(output_green))


        # Record results
        per_results = {'prompt': input_text, 
                    'output_green': output_green, 
                    'output_wo_green': output_wo_green, 
                    } 
        # print(per_results)
        output_file.write(json.dumps(per_results) + '\n') 
        print('-- save contents & settings log --')
        #torch.cuda.empty_cache()

    output_file.close()
    return None


def running_with_policy(policyfile='default.ini',log_dir = 'outputs'):
    debiasing_policy = csPolicy()
    debiasing_policy.read(f'./policy_ex/{policyfile}')
    debiasing_policy.show()
    
    args = parse_args()
    args.result_local_path = os.path.join(log_dir,f'process_{debiasing_policy.name}')
    args.result_global_path = os.path.join(log_dir,f'content_{debiasing_policy.name}')
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(args.result_local_path):
        os.mkdir(args.result_local_path)
    if not os.path.exists(args.result_global_path):
        os.mkdir(args.result_global_path) 

    main(args, debiasing_policy)





if __name__ == "__main__":
    # policy base
    policy_path = 'policy_ex'

    # running selections
    '''
    for now_policy_file in ['allrho0.ini','allrho0.ini','allrho0.ini','allrho0.ini','none.ini']:
        running_with_policy(policyfile=now_policy_file)
    '''

    # running all
    
    for root, dirs, files in os.walk(policy_path):
        for file in files:
            print(file)    
            policy_name = file
            running_with_policy(policyfile=policy_name)
            break
        break
    
    

