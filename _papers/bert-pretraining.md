---
layout: paper
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]
date: 2018-10-11
publication: "NAACL-HLT 2019"
paper_id: bert-pretraining
doi: "10.48550/arXiv.1810.04805"
url_paper: "https://arxiv.org/abs/1810.04805"
topics: [deep-learning, nlp]
tags: [transformer, BERT, NLP, pre-training, fine-tuning, language model]
reading_time: 60
progress: 75
queue: true
priority: "high"
queue_date: 2023-03-15
abstract: |
  We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

  BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
key_findings: |
  - Introduced BERT, a bidirectional language representation model based on the Transformer architecture
  - Presented a novel pre-training approach with two tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
  - Demonstrated the effectiveness of bidirectional training compared to left-to-right or shallow concatenation approaches
  - Established new state-of-the-art results on 11 NLP tasks including GLUE, SQuAD, and SWAG
  - Showed that pre-training + fine-tuning approach works well across a wide range of NLP tasks
  - Provided evidence that deeper Transformer models (BERT-large) outperform shallower ones (BERT-base)
citation:
  apa: "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 4171-4186)."
  mla: "Devlin, Jacob, et al. \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.\" Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 4171-4186."
  chicago: "Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.\" In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171-4186. 2019."
related_papers: [attention-is-all-you-need, gpt3]
---

## Summary

This paper introduces BERT (Bidirectional Encoder Representations from Transformers), a language representation model designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context. BERT significantly advanced the state-of-the-art in numerous NLP tasks and established a new paradigm for transfer learning in NLP.

## Key Innovations

### Bidirectional Training

Unlike previous models like GPT which were trained using a unidirectional language model, BERT uses masked language modeling (MLM) to enable truly bidirectional representations. This allows the model to incorporate context from both directions when making predictions.

### Pre-training Tasks

BERT uses two novel pre-training tasks:

1. **Masked Language Modeling (MLM)**: Randomly mask 15% of tokens in the input and train the model to predict the original vocabulary id of the masked word based on its context.

2. **Next Sentence Prediction (NSP)**: Train the model to understand relationships between sentences by predicting whether two segments follow each other in the original text.

### Input Representation

BERT uses a unique input representation that allows it to handle both single sentence and sentence pair tasks:

- Token embeddings: WordPiece embeddings with a 30,000 token vocabulary
- Segment embeddings: Distinguish between sentence A and sentence B
- Position embeddings: Learn position information without explicit positional encoding

## Model Variants

The paper introduces two model sizes:

- **BERT-Base**: 12 Transformer blocks, 768 hidden size, 12 attention heads (110M parameters)
- **BERT-Large**: 24 Transformer blocks, 1024 hidden size, 16 attention heads (340M parameters)

## Fine-tuning Approach

A key advantage of BERT is its simple fine-tuning procedure. The same pre-trained model can be fine-tuned for a wide variety of downstream tasks with minimal task-specific parameters:

- **Sentence Classification**: Add a classification layer on top of the [CLS] token representation
- **Sentence Pair Classification**: Process both sentences together with self-attention
- **Question Answering**: Predict the start and end positions of the answer span
- **Named Entity Recognition**: Classify each token into entity categories

## Personal Notes

I'm currently working through this paper, and I find BERT's bidirectional approach particularly interesting. The ability to incorporate context from both directions makes intuitive sense for language understanding, as humans certainly use both preceding and following context when interpreting text.

The masked language modeling task is clever - by masking random tokens, BERT learns to use context from both directions to predict words. This overcomes the limitation of traditional language models that can only be trained left-to-right or right-to-left.

BERT's impact on the field has been tremendous. It not only achieved state-of-the-art results on numerous benchmarks but also established the pre-training/fine-tuning paradigm that has become standard in NLP.

## Implementation Considerations

When implementing BERT or fine-tuning it for specific tasks:

- Batch size matters significantly, with larger batches generally producing better results
- Learning rate and number of training epochs need careful tuning for each downstream task
- Using the appropriate pooling strategy for different tasks is important
- For longer sequences, consider using a sliding window approach

## Questions for Further Research

- How important is the NSP task for BERT's performance? (Later research suggested it may be less important than originally thought)
- Can the masking strategy be improved? (Whole word masking, entity masking, etc.)
- How does BERT encode syntactic and semantic information across its layers?
- What is the optimal fine-tuning approach for different types of tasks?