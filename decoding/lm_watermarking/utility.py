import torch 

def bias_measure(input_text):
    word_list_A = ['he', 'son', 'his', 'him', 'father', 
        'man', 'boy', 'himself', 'male', 'brother', 'sons', 
        'fathers', 'men', 'boys', 'males', 'brothers', 
        'uncle', 'uncles', 'nephew', 'nephews']

    word_list_B = ['she', 'daughter', 'hers', 'her', 'mother', 
        'woman', 'girl', 'herself', 'female', 'sister', 'daughters', 
        'mothers', 'women', 'girls', 'females', 'sisters', 'aunt', 
        'aunts', 'niece', 'nieces']

    input_text = input_text.lower()
    words = input_text.split()
    
    count_A = sum(words.count(word) for word in word_list_A)
    count_B = sum(words.count(word) for word in word_list_B)
    
    # bias_score = count_A - count_B
    return {
        "count_male": count_A,
        "count_female": count_B,
        # "bias_score": bias_score
    }

if __name__ == '__main__':
    input_text = "He and his father went to the park where she and her mother were playing."
    result = bias_measure(input_text)
    print(result)