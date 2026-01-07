import time 
import functools 
import json 
from pathlib import Path
from argparse import ArgumentParser  

import requests
# import asyncio
# import aiohttp
# from aiolimiter import AsyncLimiter


import torch 
import torch.nn as nn 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoModelWithLMHead 
from transformers import GPTNeoForCausalLM
from transformers import AutoTokenizer, GPT2Tokenizer 

def parse_arg():
    parser = ArgumentParser() 
    parser.add_argument(
        "--seed", 
        type=int,
        default=1, 
        help="The random seed to ensure the reproducity"
    )
    parser.add_argument(
        "--model",
        choices=[
            "EleutherAI/gpt-neo-125M",         
        ], 
        default="EleutherAI/gpt-neo-125M",
        help="The model we used"
    )
    parser.add_argument(
        "--banheads",
        type=int,
        nargs="+",  
        default=[], 
        help="Baned heads"
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",  
        default=["valid", "test"], 
        help="runing on dataset"
    )
    parser.add_argument(
        "--targeted_bias",
        choices=[
            "characteristics",
            "ability",
            "gender_and_sex",
            "socioeconomic_class",
            "race_ethnicity",  
            "body_type",
            "cultural",
            "religion",
            "age",
            "nonce",
            "sexual_orientation",  
            "political_ideologies",  
            "nationality",
            "NaN",         
        ],
        default="gender_and_sex",
        help="The group for which biased is assessed using the holistic bias framework",
    )   

    parser.add_argument(
        "--stride",
        type=int, 
        default=512,
        help="Stride used for compute ppl"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128, # TODO 1024,
        help="Batch size for the generate",
    )
    parser.add_argument(
        "--max_continuation_length",
        type=int,
        default=40,
        help="The maximum length of the continuation",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=22,
        help="The maximum length of the prompt",
    )
    return parser.parse_args() 


def timer(func):
    @functools.wraps(func) 
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() 
        result = func(*args, **kwargs) 
        end_time = time.perf_counter() 
        elapsed = end_time-start_time
        print(f"function {func.__name__} exe time: {elapsed:4f} seconds")
        return result
    return wrapper

 
@timer
def compute_ppl(model, tokenizer, stride, max_position_embeddings):
    """
    Compute perplexity of the model. 
    Copied from https://huggingface.co/docs/transformers/perplexity
    """
    ppl = {}
    for split in ["valid", "test"]:
        names = []    
        with open("./wikitext-2-raw-v1_" + split + ".txt", 'r') as fp:
            for line in fp:
                # remove linebreak from a current name
                # linebreak is the last character of each line
                x = line
                # add current item to the list
                names.append(x) 
        encodings = tokenizer("".join(names) , return_tensors="pt")

        max_length = max_position_embeddings
        max_length=int(max_length/2) # is equal to the max_legnth/4 
        seq_len = encodings.input_ids.size(1)   #  e.g. 247289
        
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc   
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which 
                # averages over valid labels
                # N.B. the model only calculates loss over 
                # trg_len - 1 labels, because it internally shifts 
                # the labels to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            # print(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        ppl[split] = round(float(torch.exp(torch.stack(nlls).mean())), 3)

    return ppl





def analyze_toxicity(texts):
    rt = [] 
    # 配置
    API_KEY = "AIzaSyBS87gEEhUOnO2R085rDVE3whxE_sR2hG8"
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"
    for text in texts:
        payload = {
            "comment": {
                "text": text
            },
            "languages": ["en"],
            "requestedAttributes": {
                "TOXICITY": {}
            }
        }

        # 发送 POST 请求
        response = requests.post(
            url,
            json=payload,  # 自动设置 Content-Type: application/json 并序列化
            timeout=5      # 设置超时
        )

        # 检查响应
        if response.status_code == 200:
            result = response.json() 
            toxicity_score = result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            rt.append(toxicity_score)
            # print(f"Toxicity score: {toxicity_score:.3f}")
        else:
            rt.append(0)
            print(f"Error {response.status_code}: {response.text}")
        
        time.sleep(1)
    return rt 

# print(analyze_toxicity(["sb", "you idot"]))

# limiter = AsyncLimiter(max_rate=1, time_period=1)  # 1 request per second 

# async def analyze_toxicity(session, text):
#     payload = {
#         "comment": {
#             "text": text
#         },
#         "languages": ["en"],
#         "requestedAttributes": {
#             "TOXICITY": {}
#         }
#     }

#     # 获取令牌（若无可用令牌则等待）
#     async with limiter:
#         print(URL)
#         async with session.post(URL, json=payload, timeout=10) as resp:
#             if resp.status == 200:
#                 data = await resp.json()
#                 print(data)
#                 score = data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
#                 return text, score
#             else:
#                 error = await resp.text()
#                 print(f"Error for '{text}': {resp.status} - {error}")
#                 return text, None

# async def analyze_toxicity_main(): 
#     texts = ["sb", "you idot"] 
#     async with aiohttp.ClientSession() as session:
#         # 并发提交所有任务（但受 limiter 控制速率）
#         tasks = [analyze_toxicity(session, text) for text in texts]
#         results = await asyncio.gather(*tasks)

#     for text, score in results:
#         if score is not None:
#             print(f"'{text}' → Toxicity: {score:.3f}")
#         else:
#             print(f"'{text}' → Failed")


# asyncio.run(analyze_toxicity_main()) 



@timer
def gen_prompt(
    model, tokenizer, 
    data, 
    tox_model,  # do not use 
    batch_size, 
    max_continuation_length, 
    max_prompt_length
): 
    outputs = [] # , [0 for _ in range(len(data))]

    for idx in range(0, len(data), batch_size):
        batch = data[idx : idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_length=len(inputs["input_ids"][0]) + max_continuation_length,
            do_sample=True,
        )
        decoded_sequences = tokenizer.batch_decode(
            output_sequences, skip_special_tokens=True
        )   
        outputs += decoded_sequences
    toxicity_scores = analyze_toxicity(outputs)
         
    return outputs, toxicity_scores


if __name__ == "__main__": 
    args = parse_arg() 
    print(args) 
    
    torch.manual_seed(args.seed)

    # Prel: please download model and write config file first. 

    # Step 1: set model, tokenizer, config .... 
    if args.model in ["gpt2", "distilgpt2"]:
        model = AutoModelWithLMHead.from_pretrained(
            "./saved_models/cached_models/" + args.model
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + args.model, 
            padding_side="left"
        )
    elif args.model in ["EleutherAI/gpt-neo-125M"]:
        model = GPTNeoForCausalLM.from_pretrained(
            "./saved_models/cached_models/" + args.model
        ).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + args.model, 
            padding_side="left"
        )
        
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # read some config of model, such as num_heads ... 
    # different models may have different attr name, so 
    # we save in a file 
    model_configs = json.load(open("./models_config.json", "r"))
    num_heads = model_configs[args.model]["num_heads"]
    num_layers = model_configs[args.model]["num_layers"] 
    head_dim = model_configs[args.model]["head_dim"] 
    max_length = model_configs[args.model]["max_length"]
    # for multi-head attention now, TODO: Qwen3 use a different attention 

    print(f"Currently used model: {args.model}")
    print(f"max_length: {max_length}")
    print(f"num of layers: {num_layers}")
    print(f"num of heads: {num_heads}")
    print(f"head dim: {head_dim}")
    print("-"*100)
    print(model) 
    print("-"*100)
    

    # Step2: prune head or mask attention map or gated attention  
    # ban heads id list args.banheads
    idx_pruned_heads = args.banheads
    idx_pruned_heads_relative = {} # layer_id : [head_idx ... ]
    idx_pruned_layers = [int(x / num_heads) for x in idx_pruned_heads]
    for layer in list(set(idx_pruned_layers)):
        idx_pruned_heads_relative[layer] = [idx_pruned_heads[i]%num_heads for i,x in enumerate(idx_pruned_layers) if x == layer]
    print(f"Pruned heads: {idx_pruned_heads_relative}")
    
    for layer_id, heads in idx_pruned_heads_relative.items(): 
        # print(model.transformer.h[0].attn.attention.k_proj.weight) # out_features x in_features
        # print(model.transformer.h[0].attn.attention.v_proj.weight)
        # print(model.transformer.h[0].attn.attention.q_proj.weight)
        # print(model.transformer.h[0].attn.attention.out_proj.weight)
        # print(model.transformer.h[0].attn.attention.out_proj.bias)
        for head_id in heads: 
            print(f"setting {layer_id} - {head_id} to zeros...")
            start = head_dim*head_id; end = head_dim*(head_id+1); 
            model.transformer.h[layer_id].attn.attention.out_proj.weight.data[:, start:end] = 0
              
    

    # Step3: compute ppl 
    model_name = args.model.replace("/", "_")
    banheads_name = "noban" if len(args.banheads) == 0 else "_".join((str(hid) for hid in args.banheads)) 

    ppl_outputfile = f"output/{args.seed}/{model_name}_{banheads_name}/ppl.json"
    if not Path(ppl_outputfile).exists():
        ppl = compute_ppl(model, tokenizer, args.stride, int(max_length/2)) 
        Path(ppl_outputfile).parent.mkdir(parents=True, exist_ok=True)
        with open(ppl_outputfile, "w", encoding="utf-8") as f:
            json.dump(ppl, f, indent=4) 
    
    # Step4: generations and toxicity 
    for split in args.splits:
        full_results = {"generations":[]}
        prompts = json.load(
            open("./prompts/holistic/social_biases_" + split + ".json", "r")
        ) 
        for group, group_prompts in prompts.items():
            if group != args.targeted_bias:  
                continue
            prompts = group_prompts["prompts"][:args.batch_size] # List[str]  TODO: currently only use one batch
            generations, toxicity_scores = gen_prompt(
                model, tokenizer, 
                prompts, 
                None, # use google api 
                args.batch_size, 
                args.max_continuation_length, 
                args.max_prompt_length
            ) 
            full_results["generations"].extend(
                [
                    { 
                        "group": group,  
                        "prompt": prompt_text,
                        "generation": gen,
                        "toxicity_score": tox_score,  
                    }
                    for gen, prompt_text, tox_score in zip(
                        generations, prompts, toxicity_scores
                    )
                ]
            )    
        
        output_dir = f"output/{args.seed}/{model_name}_{banheads_name}/{split}_{args.targeted_bias}/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)  
        with open(output_dir + "generations.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4) 
