---
layout: paper
title: "Attention Is All You Need"
authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"]
date: 2017-06-12
publication: "Advances in Neural Information Processing Systems (NIPS 2017)"
paper_id: attention-is-all-you-need
doi: "10.48550/arXiv.1706.03762"
url_paper: "https://arxiv.org/abs/1706.03762"
topics: [deep-learning, nlp]
tags: [transformer, attention mechanism, sequence models, neural networks, NLP]
reading_time: 45
progress: 100
abstract: |
  The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
key_findings: |
  - Introduced the Transformer, a novel architecture based entirely on self-attention mechanisms without using recurrence or convolution
  - Demonstrated superior performance on machine translation tasks compared to previous state-of-the-art models
  - Required significantly less training time due to increased parallelization capabilities
  - Introduced multi-head attention, which allows the model to jointly attend to information from different representation subspaces
  - Established new state-of-the-art results on the WMT 2014 English-to-German and English-to-French translation benchmarks
  - Proved the architecture generalizes well to other tasks like English constituency parsing
citation:
  apa: "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008)."
  mla: "Vaswani, Ashish, et al. \"Attention Is All You Need.\" Advances in Neural Information Processing Systems, 2017, pp. 5998-6008."
  chicago: "Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. \"Attention Is All You Need.\" In Advances in Neural Information Processing Systems, 5998-6008. 2017."
related_papers: [bert-pretraining, gpt3]
---

## Summary

This groundbreaking paper introduced the Transformer architecture, which has become the foundation for most modern NLP models. The authors presented a neural network architecture based entirely on attention mechanisms, completely removing the need for recurrence or convolution operations that were previously standard in sequence models.

## Key Innovations

### Self-Attention Mechanism

The paper's most significant contribution is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when encoding each word. This enables the model to capture long-range dependencies more effectively than RNNs or CNNs.

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

Where Q, K, and V are query, key, and value matrices derived from the input sequence.

### Multi-Head Attention

Instead of performing a single attention function, the authors proposed using multiple attention heads in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

This allows the model to jointly attend to information from different representation subspaces at different positions.

### Positional Encoding

Since the Transformer doesn't use recurrence or convolution, it has no inherent understanding of the order of words in a sequence. To address this, the authors added positional encodings to the input embeddings:

```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

## Personal Notes

The Transformer architecture represents a paradigm shift in sequence modeling. By removing the sequential computation requirement of RNNs, it allows for much greater parallelization during training. This has enabled training on much larger datasets, which has been crucial for the development of models like BERT and GPT.

The attention mechanism is intuitive yet powerful - it allows the model to focus on relevant parts of the input sequence when making predictions. This is somewhat analogous to how humans process language, focusing on relevant words and their relationships rather than processing every word with equal importance.

The paper is remarkably well-written and accessible despite introducing a complex architecture. The authors clearly explain their motivation, methodology, and results, making it easier to understand the significance of their contribution.

## Implementation Challenges

When implementing the Transformer, pay close attention to:
- Layer normalization placement (pre-norm vs. post-norm)
- Proper initialization of parameters
- Learning rate scheduling (the authors used a custom schedule with a warmup period)
- Label smoothing during training

## Impact and Applications

The Transformer architecture has become the foundation for most state-of-the-art NLP models including:
- BERT, RoBERTa, and other bidirectional encoders
- GPT series of generative models
- T5 and other encoder-decoder models

It has also been applied to:
- Computer vision (Vision Transformer)
- Audio processing
- Reinforcement learning
- Protein structure prediction (AlphaFold)