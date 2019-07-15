## Trainable Korean spacing (TaKos)

Simply create your own Korean spacing model with your sentence text file!

### Requirements

- Python 3.6.2

```bash
## versions will be update soon
pip install -r requirements
```

### How to Run

#### Create Spacing Agent
```python
from takos.agents.word_segment import WordSegmentAgent
spacing_agent = WordSegmentAgent('your_config.json')
```

#### Train 
```python
spacing_agent.train()
```

#### Eval
```python
spacing_agent.eval()
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
