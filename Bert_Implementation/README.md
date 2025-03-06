## Bert Inference Implementation

## Introduction
BERT (Bidirectional Encoder Representations from Transformers) leverages a transformer-based neural network to understand and generate human-like language. BERT employs an encoder-only architecture. In the original Transformer architecture, there are both encoder and decoder modules. The decision to use an encoder-only architecture in BERT suggests a primary emphasis on understanding input sequences rather than generating output sequences.

## Folder Structure


```
Model_Evaluation /
└── Bert_Implementation /
    ├── reference /
    │   ├── bert.py                # Bert model implementation
    ├── tests/
    │   ├── test_bert.py           # Unit tests for Bert model
    │   ├── test_bert_demo.py      # Whole Model test setup
    └── README.md                     # Documentation for Bert module
```


# Details

- The reference model is implemented in Bert_Implementation/reference/bert.py by adapting the Bert model from the torchvision library, but without the dependencies on torchvision.
- Each sub-module of this reference model is validated against the original torchvision package using the Pearson Correlation Coefficient (PCC) metric to ensure accuracy.
- Unit tests for sub-modules are located in `Bert_Implementation/tests/test_bert.py` for input context and question.
- A demo setup that tests the entire model can be found in `Bert_Implementation/tests/test_bert_demo.py`.
- Pretrained weights are used from the actual  `BertForQuestionAnswering` - `bert-large-uncased-whole-word-masking-finetuned-squad`

## Commands to Run



### To run all the tests with logging output:
`pytest Model_Evaluation/Bert_Implementation/`

This will return both the test results.

### To run the demo test alone:
`pytest Model_Evaluation/Bert_Implementation/tests/test_bert_demo.py -s` 

### To run sub-module tests alone:
`pytest Model_Evaluation/Bert_Implementation/tests/test_bert.py`

## Expected Demo Results

- Custom BERT Answer: j. k. rowling
- Pre-trained BERT Answer: j. k. rowling

