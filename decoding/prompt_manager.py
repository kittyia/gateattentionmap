

import json
from policy_manager import csPolicy
from tqdm import tqdm


debiasing_policy = csPolicy()
debiasing_policy.read(r'policy_ex/default.ini')
debiasing_policy.show()

data_path = 'data/C4/sampled_dataset_train.json'
with open(data_path, "r") as input_file:
        data = json.load(input_file)
        # print(data)
        for input_text in tqdm(data):
            print(input_text)


# run without policy
