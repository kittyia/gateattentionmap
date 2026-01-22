# 目录

- decoding: 赵老师早期发给我们的代码，暂时不用管
- fairness and bias paper: 我精读过的一些paper，暂时不用管
- readme_images: readme.md 用到的图片
- prompts/holistic: 数据
    + social_biases_valid_groups.json： axis是每个prompts属于的大类，bucket是每个prompts属于的子类 
    + social_biases_valid.json： prompts数据 

# 文件

- wikitext-2-raw-v1_valid/test/.txt: 算ppl要用的
- models_config.json: 一些模型的参数信息 head layers
- try.ipynb: decoding策略的尝试
- test.ipynb: 不用管
- maskattentionmap.py 主要python脚本文件
- maskattentionmap.ipynb 主要的流程文件 


总结一下：从maskattentionmap.ipynb maskattentionmap.py读，跑通就可以。

# 复现流程

1. 运行maskattentionmap.ipynb block1，下载模型，分词器保存到本地
2. 运行maskattentionmap.ipynb block2，把要用到的模型信息存到models_config.json（模型有多少层，多少个头，每个头的维度，最大的输入token长度）
3. maskattentionmap.ipynb block3 是用来create scripts的，后面你提交任务到分布节点上可能需要用，现在可以不管。
4. 接下来跑maskattentionmap.py运行实验就可以了

先板书讲一下我们干了什么，代码怎么实现的。


最后是如何调用：
> `python maskattentionmap.py --seed 1 --model EleutherAI/gpt-neo-125M --banheads {headid} {headid2} --splits valid --targeted_bias gender_and_sex`




