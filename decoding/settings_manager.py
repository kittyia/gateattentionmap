import argparse

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")
    
    # -large languate model setting
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        #default="facebook/opt-1.3b",
        default="facebook/opt-125m",
        # default="meta-llama/Llama-3.1-8B",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )

    # -prompt setting
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )

    # periods setting
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximmum number of new tokens to generate.",
    )


    parser.add_argument(
        "--generation_seed",
        type=int,
        default=1,
        help="Seed for setting the torch global rng prior to generation.",
    )

    # generating mulitinomial sampling
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.0,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )

    # calculation settings
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default='0',
    )

    #
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )

    # watermark settings
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=20.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )

    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    ####
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default='/data/mjmao/ood/hf_models',
        help="input model cache dir",
    )

    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default='data/C4/sampled_dataset_train.json',
        help="input prompts data file",
    )

    # # # #
    parser.add_argument(
        "--result_local_path",
        type=str,
        default='./contents',
        help="save path of in-process logs",
    )

    parser.add_argument(
        "--result_global_path",
        type=str,
        default='./contents',
        help="save path of content logs",
    )

    parser.add_argument(
        "--policy_ex_path",
        type=str,
        default='./policy_ex',
        help="save path of ex policies",
    )

    args = parser.parse_args()
    return args

if __name__ =='__main__':
    test = parse_args()
    print(test)