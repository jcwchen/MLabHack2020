# Bert Squad

### Github Link
```https://github.com/jcwchen/MLabHack2020/edit/master/BertSquad/```

### Onnx File Name
```bertsquad-8```

### Include Training and Inference
```False```

### Model Name
```bert-squad```

### Category Name
```Natural Language Processing```

### Tasks
```Machine Comprehension, Query Answering, Text Understanding, Sentence Relation```

### Cover Image
```https://miro.medium.com/max/1840/1*QhIXsDBEnANLXMA0yONxxA.png```

### Input Description
The input is a paragraph and questions relating to that paragraph. The model uses the WordPiece tokenization method to split the input paragraph and questions into list of tokens that are available in the vocabulary (30,522 words). Then converts these tokens into features

- input_ids: list of numerical ids for the tokenized text
- input_mask: will be set to 1 for real tokens and 0 for the padding tokens
- segment_ids: for our case, this will be set to the list of ones
- label_ids: one-hot encoded labels for the text

### Preprocessing Description
Write an inputs.json file that includes the context paragraph and questions.

```
%%writefile inputs.json
{
  "version": "1.4",
  "data": [
    {
      "paragraphs": [
        {
          "context": "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space",
          "qas": [
            {
              "question": "where is the businesses choosing to go?",
              "id": "1"
            },
            {
              "question": "how may votes did the ballot measure need?",
              "id": "2"
            },
            {
              "question": "By what year many Silicon Valley businesses were choosing the Moscone Center?",
              "id": "3"
            }
          ]
        }
      ],
      "title": "Conference Center"
    }
  ]
}
```
Get parameters and convert input examples into features
```python
# preprocess input
predict_file = 'inputs.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer,
                                                                              max_seq_length, doc_stride, max_query_length)
```

### Output Description
For each question about the context paragraph, the model predicts a start and an end token from the paragraph that most likely answers the questions.

### Postprocessing Description
Write the predictions (answers to the questions) in a file.
```python
# postprocess results
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
write_predictions(eval_examples, extra_data, all_results,
                  n_best_size, max_answer_length,
                  True, output_prediction_file, output_nbest_file)
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
```Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova```

### Paper Link
```https://arxiv.org/pdf/1810.04805.pdf```

### Evaluation Metrics
```Exact Matching (EM)```

### Evaluation Results
```80.7```

### Training/Validation Loss Graph
```N/A```
