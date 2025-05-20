---
layout: paper
title: "Language Models are Few-Shot Learners"
authors: ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell", "Sandhini Agarwal", "Ariel Herbert-Voss", "Gretchen Krueger", "Tom Henighan", "Rewon Child", "Aditya Ramesh", "Daniel M. Ziegler", "Jeffrey Wu", "Clemens Winter", "Christopher Hesse", "Mark Chen", "Eric Sigler", "Mateusz Litwin", "Scott Gray", "Benjamin Chess", "Jack Clark", "Christopher Berner", "Sam McCandlish", "Alec Radford", "Ilya Sutskever", "Dario Amodei"]
date: 2020-05-28
publication: "Advances in Neural Information Processing Systems (NeurIPS 2020)"
paper_id: gpt3
doi: "10.48550/arXiv.2005.14165"
url_paper: "https://arxiv.org/abs/2005.14165"
topics: [deep-learning, nlp]
tags: [GPT, language models, few-shot learning, zero-shot learning, NLP, scaling laws]
reading_time: 90
progress: 50
queue: true
priority: "medium"
queue_date: 2023-04-02
abstract: |
  Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.
key_findings: |
  - Demonstrated that scaling up language models significantly improves few-shot and zero-shot learning capabilities
  - Introduced GPT-3, a 175 billion parameter autoregressive language model, the largest of its kind at publication
  - Showed that larger models exhibit qualitatively different behavior: they can perform tasks via "in-context learning" without explicit fine-tuning
  - Established that model performance scales as a power-law with model size, dataset size, and amount of computation
  - Achieved strong results across many NLP tasks without any gradient updates or fine-tuning
  - Identified limitations including struggles with some reasoning tasks and potential biases from web training data
  - Raised important ethical considerations about potential misuse and the economics of large language models
citation:
  apa: "Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877-1901."
  mla: "Brown, Tom B., et al. \"Language Models are Few-Shot Learners.\" Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 1877-1901."
  chicago: "Brown, Tom B., Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, et al. \"Language Models are Few-Shot Learners.\" In Advances in Neural Information Processing Systems, vol. 33, 1877-1901. 2020."
related_papers: [attention-is-all-you-need, bert-pretraining]
---

## Summary

This landmark paper introduced GPT-3, a 175 billion parameter language model that demonstrated unprecedented few-shot learning capabilities. The authors showed that by significantly scaling up model size, training data, and compute, language models can perform a wide range of tasks without task-specific fine-tuning - simply by providing a few examples in the prompt (few-shot learning) or even just a natural language instruction (zero-shot learning).

## Key Innovations

### Scale as a Path to General-Purpose Models

The paper's central thesis is that scale leads to qualitatively different model capabilities. By increasing model size by 10x compared to previous models, GPT-3 achieves strong performance across diverse tasks without any gradient updates or fine-tuning.

### In-Context Learning

GPT-3 introduced the concept of "in-context learning" where the model learns to perform tasks from a few demonstrations provided in the context window, without weight updates. This mimics how humans can quickly adapt to new tasks given a few examples.

### Learning Modalities

The paper explores several learning settings:

1. **Few-shot learning**: Providing K examples of the task in the prompt (typically K=10-100)
2. **One-shot learning**: Providing just one example in the prompt
3. **Zero-shot learning**: Providing only a natural language instruction with no examples

### Scaling Laws

The authors observe consistent scaling behaviors:
- Performance improves smoothly as a power-law with model size
- Larger models are more sample-efficient learners
- Larger models continue to improve even when data and compute are scaled proportionally

## Model Variants

The paper evaluates 8 different model sizes:
- Smallest: 125 million parameters
- Largest (GPT-3): 175 billion parameters

All models use the same architecture: an autoregressive Transformer with modified initialization, pre-normalization, and reversible tokenization.

## Results Across Tasks

GPT-3 was evaluated on a wide range of tasks including:

- **Translation**: Competitive results on WMT14 benchmarks
- **Question answering**: Strong performance on TriviaQA without fine-tuning
- **Cloze and completion tasks**: Near state-of-the-art on multiple datasets
- **Common sense reasoning**: Significant improvements on Winograd schemas
- **Reading comprehension**: Competitive with fine-tuned models on several benchmarks
- **Natural language inference**: Mixed results depending on the framing
- **Word scrambling and manipulation**: Novel tasks showing in-context learning capabilities
- **Arithmetic**: Ability to perform 3-digit addition, subtraction, and multiplication

## Personal Notes (Partial Reading)

I'm currently halfway through this paper, and it's truly remarkable how scaling up transformed what was possible with language models. The concept of in-context learning feels like a genuine step toward more general AI systems.

The few-shot learning results are particularly impressive - the model can adapt to new tasks given just a handful of examples in the prompt, without any weight updates. This mimics human learning much more closely than traditional fine-tuning approaches.

What stands out most to me so far is how GPT-3's capabilities emerge organically from scale, rather than being explicitly designed. This emergent behavior supports the scaling hypothesis - that many capabilities can be achieved simply by scaling up existing architectures.

## Limitations and Concerns

The paper candidly discusses several limitations:

- Struggles with tasks requiring multi-step reasoning
- Sample inefficiency compared to humans
- Sensitivity to prompt formatting and wording
- Potential to generate harmful, biased, or misleading content
- Economic and environmental costs of training such large models
- Limited understanding of what the model actually learns

## Questions to Explore Further

- How can we better understand what knowledge is encoded in these large models?
- Can the sample efficiency of in-context learning be improved?
- What are the limitations of the scaling hypothesis?
- How can we address the ethical concerns raised by models like GPT-3?
- What architectural innovations might complement pure scaling approaches?

I'll continue reading the paper to better understand the ethical considerations and broader impacts the authors discuss in the latter sections.