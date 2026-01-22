import json
import random
from tqdm import tqdm
from utility import bias_measure

if __name__ == '__main__':
    # Load the input data
    with open('data/C4/sampled_dataset_train.json', "r") as input_file:
        data = json.load(input_file)
    
    number = 0
    sampled_data = []
    for input_text in tqdm(data):
        if number == 64:
            break
        input_text = (input_text)
        result = bias_measure(input_text)
        if result['count_male'] + result['count_male'] >= 3:
            sampled_data.append(input_text)
            number += 1
        
    # # Sample 64 random input texts
    # sampled_data = random.sample(data, 64)
    
    # Save the sampled data into another JSON file
    with open('data/C4/sampled_64_subset.json', "w") as output_file:
        json.dump(sampled_data, output_file, indent=4)

    print("Successfully sampled 64 input texts and saved to data/C4/sampled_subset.json")
