## Trainable Korean spacing (TaKos)

Simply create your own Korean spacing model with your sentence text file!

### Install 

This is an alpha version. It will be installed by pip install in the future.

```bash
git clone https://github.com/Taekyoon/takos-alpha.git
pip install -r requirements
python setup.py install
```

### Requirements

- Python 3.6.2

```bash
## versions will be update soon
pip install -r requirements
```

### How to Run

#### Create Spacing Agent
Currently, I designed this library configs using JSON file, 
so you need to input `configs.json` path to agent object as a parameter.

JSON sample configs are in `scripts` folder, and you can apply these files to create `WordSegmentAgent` object.
To test your spacing model, I added 1,100 sample sentences which are sampled from Korean Wiki dataset.
The model will not perform well, but you will understand how this agent works.

```python
from takos.agents.word_segment import WordSegmentAgent
spacing_agent = WordSegmentAgent('your_config.json')
```

Once you want to input your sentence file to spacing model, 
you have to configure JSON parameter `"dataset"` on your `configs.json`.
Here are the variables of dataset configs.

```json
{
  ......
    "dataset": {
        "name": "your_dataset",
        "train": {
          "vocab_min_freq": 10,
          "input": "./samples/train.txt"
        },
        "test": {
          "limit_len": 150,
          "input": "./samples/test.txt"
        }
      }
  ......
}
```

When you tune those configuration as you like, your model is ready to train and evaluate and run!

#### Train
The train method is performed by configs file, and those are the variables you need to consider.
```json
{
  ......
    
    "train": {
        "epochs": 100,
        "eval_steps": -1,
        "learning_rate": 3e-4,
        "eval_batch_size": 10,
        "batch_size": 64,
        "sequence_length": 50
      }
}
```
 
```python
spacing_agent.train()
```

#### Eval
```python
spacing_agent.eval()

''' Results
+-----------+--------+
|    Name   | Score  |
+-----------+--------+
| WER score | 0.4896 |
| SER score | 0.8200 |
|  F1 score | 0.8925 |
| ACC score | 0.8950 |
+-----------+--------+
'''
```

#### Run spacing
```python
spacing_agent('학교종이땡땡땡')

''' Results

{'input': '학교종이땡땡떙',
 'label': ['B', 'I', 'I', 'E', 'B', 'I', 'E'],
 'sequence_score': array([12.990488], dtype=float32),
 'output': '학교종이 땡땡떙',
 'segment_pos': [0, 4]}
 
'''
```

### Contact Me!!

Still, this is an alpha version library. 
So if you have any issues while using this, feel free to contact by writing issues.
