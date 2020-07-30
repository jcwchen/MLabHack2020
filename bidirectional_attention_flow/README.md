# Bidirectional Attention Flow

### Github Link
```https://github.com/jcwchen/MLabHack2020/edit/master/bidirectional_attention_flow/```

### Onnx File Name
```bidaf-9```

### Include Training and Inference
```False```

### Model Name
```bidirectional attention flow model```

### Category Name
```Natural Language Processing```

### Tasks
```Machine Comprehension, Query Answering, Text Understanding```

### Cover Image
```https://raw.githubusercontent.com/GauthierDmn/question_answering/master/bidaf-architecture.png```

### Input Description
Tokenized strings of context paragraph and query.

### Preprocessing Description
Tokenize words and chars in string for context and query. The tokenized words are in lower case, while chars are not. Chars of each word needs to be clamped or padded to list of length 16. Note NLTK is used in preprocess for word tokenize.

- context_word: [seq, 1,] of string
- context_char: [seq, 1, 1, 16] of string
- query_word: [seq, 1,] of string
- query_char: [seq, 1, 1, 16] of string

The following code shows how to preprocess input strings:
```python
import numpy as np
import string
from nltk import word_tokenize

def preprocess(text):
   tokens = word_tokenize(text)
   # split into lower-case word tokens, in numpy array with shape of (seq, 1)
   words = np.asarray([w.lower() for w in tokens]).reshape(-1, 1)
   # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
   chars = [[c for c in t][:16] for t in tokens]
   chars = [cs+['']*(16-len(cs)) for cs in chars]
   chars = np.asarray(chars).reshape(-1, 1, 1, 16)
   return words, chars

# input
context = 'A quick brown fox jumps over the lazy dog.'
query = 'What color is the fox?'
cw, cc = preprocess(context)
qw, qc = preprocess(query)
```

### Output Description
The model has 2 outputs.

- start_pos: the answer's start position (0-indexed) in context,
- end_pos: the answer's inclusive end position (0-indexed) in context.

### Postprocessing Description
Post processing and meaning of output.
```python
# assuming answer contains the np arrays for start_pos/end_pos
start = np.asscalar(answer[0])
end = np.asscalar(answer[1])
print([w.encode() for w in cw[start:end+1].reshape(-1)])
```
For this testcase, it would output
```
[b'brown'].
```
### Hyperparameter Description
```
N/A
```

### Dataset Name
```SQuAD v1.1```

### Dataset URL
```https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/```

### Paper Authors
```Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi```

### Paper Link
```https://arxiv.org/pdf/1611.01603.pdf```

### Evaluation Metrics
```Exact Matching (EM)```

### Evaluation Results
```68.1```

### Training/Validation Loss Graph
```N/A```
