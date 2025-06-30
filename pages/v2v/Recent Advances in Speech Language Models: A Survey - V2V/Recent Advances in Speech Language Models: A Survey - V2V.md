---
layout: page
title: "Recent Advances in Speech Language Models: A Survey"
permalink: /dm/speech-language-models-survey/
---

[Paper Link](https://arxiv.org/abs/2410.03751v1)

## Components

### 1. Speech Tokenizer

#### Semantic Understanding Objective
Speech tokenizers designed with a semantic understanding objective aim to convert speech waveforms into tokens that accurately capture the content and meaning of the speech. These tokenizers focus on extracting semantic features from the waveforms, which enhances tasks like ASR. A semantic understanding speech tokenizer typically comprises a speech encoder and a quantizer, where the speech encoder encodes the essential information from the waveform and the quantizer discretizes continuous representations into discrete tokens. Let fE(·) denote the speech encoder parameterized by θfE, we have v = fE(a; θfE), where v = (v1, v2, . . . , vP) represents the encoded representations. Since v is still continuous, a quantizer d(·) is utilized to discretize the representation. Depending on different design choices, the discrete speech tokens s = (s1, s2, . . . , sP) can either be derived from a or v. Therefore, we have s = d(v; θd) or s = d(a; θd). After that, s can be used to train the speech tokenizer as a target label (such as masking amask ⊂ a and reconstructing its corresponding label smask ⊂ s [Hsu et al., 2021]) or to train the following language model.

The key design choices lie in how to effectively encode and quantize the speech into discrete tokens.

- Wav2vec 2.0 [Baevski et al., 2020] uses a convolutional encoder followed by a product quantization module [Jegou et al., 2010] to discretize the continuous waveform. Then, a portion of the quantized representations is masked and modeled using a contrastive loss.

- W2v-BERT [Chung et al., 2021] is built upon wav2vec 2.0 and proposes to use Masked Language Modeling (MLM) loss [Devlin, 2018] in addition to contrastive loss.

- HuBERT [Hsu et al., 2021] uses the k-means algorithm to cluster the speech utterances into a certain number of hidden units, and then perform MLM to predict the target hidden units from the masked speech utterances.

- Google USM [Zhang et al., 2023c] utilizes text-injection loss [Chen et al., 2022b] at the second pre-training stage to improve the performance and robustness of the downstream tasks.

- WavLM [Chen et al., 2022a] adds the speech denoising objective during pre-training. While the majority of speech tokenizer studies focus on semantic-related tasks such as ASR and TTS, WavLM shows that speech denoising can boost the performance of non-semantic tasks such as speaker verification and speech separation.

#### Acoustic Generation Objective
Speech tokenizers with an acoustic generation objective focus on capturing the acoustic features necessary for generating high-quality speech waveforms. These tokenizers prioritize the preservation of essential acoustic characteristics over semantic content, making them suitable for speech synthesis tasks.

To generate high-quality speech waveforms, acoustic generation speech tokenizers employ a speech synthesis or speech reconstruction objective. To achieve this, the architecture typically includes an encoder, a quantizer, and a decoder. Same as before, the encoder fE(·) and quantizer d(·) transform the original waveform into discrete tokens. After that, the decoder fD(·) reconstructs these tokens back into speech waveforms. This process is represented by â = fD(s; θfE), where â is the generated or reconstructed waveform.

Neural audio codecs are very suitable for and are primarily employed as acoustic generation speech tokenizers. These codecs utilize the advanced modeling capabilities of deep neural networks to compress audio signals into a compact representation, typically in the form of quantized tokens.

For example, both SoundStream [Zeghidour et al., 2021] and EnCodec [Defossez et al., 2022] use convolution blocks as the encoder and use Residual Vector Quantization (RVQ) [Zeghidour et al., 2021] as the quantizer. This mechanism allows for codecs to efficiently transmit or store audio with minimal loss in quality. Since the output of codecs is in discrete format, they can also be leveraged by SpeechLM to autoregressively generate speech.

#### Mixed Objective
Speech tokenizers with a mixed objective aim to balance both semantic understanding and acoustic generation. Currently, the development of these tokenizers is in its early stages. Most existing mixed speech tokenizers primarily adopt the architecture of acoustic generation speech tokenizers and focus on distilling information from semantic tokenizers into the acoustic tokenizer.

- SpeechTokenizer [Zhang et al., 2023b] utilizes the RVQ-GAN [Defossez et al., 2022; Zeghidour et al., 2021] architecture, distilling semantic information from HuBERT [Hsu et al., 2021] to the first layer of RVQ.

- Building on SpeechTokenizer, Mimi [Defossez et al., 2024] employs a single vector quantizer (VQ) to extract information from WavLM [Chen et al., 2022a] and incorporates another RVQ module to learn the acoustic information.

### 2. Language Model
Due to the success of TextLMs, most SpeechLMs follow their architectures. They primarily employ transformers or decoder-only architectures (such as OPT, LLaMA) to generate speech in an autoregressive manner.

### 3. Token-to-Speech Synthesizer (Vocoder)
After the tokens have been autoregressively generated by the language model component, a token-to-speech module, often known as vocoder, is utilized to synthesize all the speech tokens back into speech waveforms.

- Direct synthesis is the pipeline where the vocoder directly converts speech tokens generated by the language model into audio waveforms.

- Input-enhanced synthesis employs an additional module to transform the tokens into a continuous latent representation before they are fed into the vocoder.

## Training Recipe

Following TextLMs, the training process for SpeechLMs can be divided into three stages: pre-training, instruction-tuning, and alignment. However, to our knowledge, there is currently no research specifically focused on the alignment process following instruction tuning. Therefore, we only discuss the works related to the pre-training and instruction-tuning stages of SpeechLMs.

### Training Stages

#### Language Model Pre-training
This phase typically involves training the language model to autoregressively predict the next token on a large corpus of speech tokens. The primary objective during this stage is to learn the statistical patterns and dependencies inherent in the speech data, enabling the model to predict the next token in a sequence based on the preceding context.

##### Training data
- ASR
  - Librispeech: An ASR corpus based on public domain audio books
  - [Libri-Light](https://www.semanticscholar.org/paper/Libri-Light%3A-A-Benchmark-for-ASR-with-Limited-or-No-Kahn-Rivi%C3%A8re/f59c038dee828e0a8c2fc28130d12e39ee4952d6)
  - [Paper 1](https://arxiv.org/abs/2111.09344)
  - [Paper 2](https://arxiv.org/abs/2101.00390)

- TTS
  - [Paper](https://arxiv.org/abs/1904.02882)

- ST
  - [Paper 1](https://arxiv.org/abs/2201.03713)
  - [Paper 2](https://arxiv.org/abs/2101.00390)

- Podcasts
  - [Paper](https://aclanthology.org/2020.coling-main.519/)

- Dialogues
  - [Paper](https://aclanthology.org/L04-1500/)

##### Initialization of Language Model

###### Cold Initialization
Model parameters are initialized randomly.

- The pioneering SpeechLM—GSLM [Lakhotia et al., 2021]—trained a transformer [Vaswani, 2017] from scratch to serve as the language model. This study demonstrated the effectiveness of the SpeechLM pipeline and compared performance across various speech tokenizer options. They found that HuBERT [Hsu et al., 2021] outperformed CPC [Oord et al., 2018] and wav2vec 2.0 [Baevski et al., 2020] in understanding speech content and generating natural speech.

- SUTLM [Chou et al., 2023] also uses a transformer as the language model. They studied the critical problem of jointly modeling speech and text tokens by comparing four different modeling methods: speech-only, text-only, concatenated speech-text, and alternating (interleaving) speech-text. They showed that the setting of alternating speech-text performs the best in cross-modal evaluations.

Different architecture from the standard transformer:

- pGSLM [Kharitonov et al., 2021] proposes a multi-stream transformer language model (MSTLM) that takes multiple streams of input and predicts multiple streams of output to generate speech units, duration, and pitch embeddings simultaneously.

- dGSLM [Nguyen et al., 2023b] introduced a dialogue transformer language model (DLM) to jointly model the dialogue speech data from the two speakers.

- To enable the listening ability of SpeechLMs while speaking, LSLM [Ma et al., 2024] proposes to attach a streaming self-supervised learning (SSL) Encoder to an autoregressive token-based TTS Model.

- Viola [Wang et al., 2024] introduced a multi-task auto-regressive codec language model to autoregressively generate codec tokens instead of speech unit tokens.

###### Continued Pre-Training
Initializing the language model with pre-trained weights from a TextLM and then adapting it to handle speech tokens.

- Research by [Hassid et al., 2024] found that starting with a textually pre-trained language model (OPT [Zhang et al., 2022b] and LLaMA [Touvron et al., 2023a]) can enhance the model's convergence rate and significantly improve its speech understanding capabilities. They also demonstrated that while training from text-pretrained checkpoints outperforms cold initialization, training from image-pretrained checkpoints yields poorer results compared to cold initialization. This indicates that not all pre-trained checkpoints are equally effective.

- AudioPaLM [Rubenstein et al., 2023] trained the SpeechLM using PaLM and PaLM-2 [Chowdhery et al., 2023; Anil et al., 2023b], showing that the SpeechLM benefits from both an increased size of the pre-trained checkpoint and a larger training dataset. The performance of SpeechLMs can be further enhanced by aligning the text and speech modality representations.

- SPIRIT-LM [Nguyen et al., 2024] found that continually pretraining on TextLM checkpoints using interleaving text and speech tokens can significantly boost the model's performance on speech understanding and generation. Additionally, their visualizations demonstrate that the similarity between text and speech features is notably higher in models trained with interleaved token sequences compared to those trained without this approach.

- AudioChatLlama [Fathullah et al., 2024] aims to ensure that the model produces consistent outputs regardless of whether the input is text or speech. They address this challenge by treating text data in ASR datasets as prompts, allowing LLaMA to generate the corresponding responses. Consequently, both text and speech versions of the prompt can be utilized to train the model to provide the appropriate response.

- Spectron [Nachmani et al., 2023] solves the text-speech representation alignment problem by jointly supervising multiple objectives. Specifically, the input speech prompt is first transcribed into its text tokens, and then the model predicts the text token response. Finally, the text response is synthesized to output speech.

#### Language Model Instruction-Tuning
Instruction-tuning refers to the process of fine-tuning SpeechLMs to follow specific instructions to perform a wide range of tasks. This phase is crucial for enhancing the pre-trained model's generalization capabilities and making it more adaptable to diverse applications.

- SpeechGPT [Zhang et al., 2023a] and SpeechGPT-Gen [Zhang et al., 2024] propose a two-stage instruction-tuning, including cross-modal instruction fine-tuning and chain-of-modality instruction fine-tuning.

  - In the first stage, instruction data are generated based on ASR datasets by appending the instruction to paired ASR data, asking the model to convert speech into text. Similarly, paired data is also used to create instruction data for performing TTS.

  - In the second stage, they construct a speech-in-speech-out dataset by transforming a text-based instruction-following dataset using TTS.

- Llama-Omni [Fang et al., 2024] also creates instruction-following data by synthesizing text-based datasets, adhering to specific constraints.

  - First, they transform the input text prompt into a format that mimics natural speech patterns.

  - Next, they discard the original text response and employ a TextLM to generate answers to the converted prompts, ensuring these responses also follow natural speech patterns.

  - Finally, they synthesize the prompt/response pairs using TTS.

- COSMIC [Pan et al., 2023] constructed speech QA data by asking GPT-3.5 to generate question-answer pairs based on the transcriptions of English TED talk speeches. They showed the model trained on their proposed speech QA dataset can generalize to unseen tasks such as speech-to-text translation using in-context learning.

## Speech Generation Paradigm

### Real-time Interaction
Refers to the capability of SpeechLMs to engage with users instantaneously.

1. User Interruption: SpeechLMs should be able to be interrupted by users and should respond appropriately to new instructions provided during the conversation.

2. Simultaneous Response: SpeechLMs should be capable of generating responses while the user is still speaking.

Require the model to effectively process input and produce output simultaneously.

- dGSLM [Nguyen et al., 2023b] introduces a dual-transformer architecture to model two-speaker dialogues, using one transformer to handle speech from each speaker. A cross-attention transformer layer is included to capture the interactions between the speakers' content.

- LSLM [Ma et al., 2024] proposes a different approach, utilizing a single decoder-only Transformer to model one speaker's speech in the dialogue. This model incorporates a streaming SSL encoder that continuously processes input from the listening channel and fuses its embeddings with those from the speaking channel.

### Silence Mode
State in which the SpeechLMs remain inactive or silent during periods of non-interaction.

- This mode is essential for creating a natural conversational flow, allowing the model to avoid unnecessary interruptions.

- It is crucial for situations where a small group of users is having a discussion, as the SpeechLM needs to discern when to join in and when to stay silent.

- Additionally, it is important for the model to learn when to disregard instructions when users are not speaking at it.

- VITA [Fu et al., 2024] is currently the only work that integrates silence mode. This method involves training the model on both query speech and non-query audio, which may include environmental sounds or non-query speech. As a result, the model learns to output the end-of-sequence token to terminate its response when non-query audio is detected.

## Downstream Applications
SpeechLMs function as generative foundation models with many primary downstream applications.

## Evaluation

### Automatic (Objective) Evaluation

#### Representation Evaluation
It refers to how input data, such as speech or text, is transformed into a format that the model can understand and process.

- Effective representation lays a solid foundation for models to understand lexical, syntax, and contextual information, which are vital for generating coherent and contextually relevant outputs.

- In the context of SpeechLMs, representation evaluation focuses on how well the model encodes speech features into meaningful vectors.

- GSLM [Lakhotia et al., 2021] uses between-speaker ABX score to measure the embedding similarity. It quantifies how well-separated the phonetic categories are. Specifically, It works by comparing three sound samples: two from the same category (A) and one from a different category (B). The test measures how often the system correctly identifies that two sounds from category A are more similar to each other than one sound from A is to a sound from B.

- Another way of evaluating representations is through speech resynthesis [Lakhotia et al., 2021]. Specifically, an input speech is encoded into tokens and then synthesized back to speech. Then, word error rate (WER) or character error rate (CER) can be computed on the ASR results of the input and resynthesized speech. This measures the information loss caused by discretizing the input speech into speech tokens, thereby evaluating the robustness of the latent representations.

#### Linguistic Evaluation
Linguistics, including lexical, syntactic, and semantic evaluation methods, assess the model's ability to generate and understand the rules for constructing words, sentences, and meaningful contents.

- These evaluations focus on the correctness and appropriateness of word choices, the grammatical structure of the outputs, and the coherence and relevance of the generated content.

- In terms of benchmark datasets:
  - sWUGGY [Nguyen et al., 2020] assesses at the lexical level by determining if the model can distinguish a real word from a (real, non-real) word pair.
  - sBLIMP [Nguyen et al., 2020] evaluates at the syntactic level by determining if the model can identify the grammatically correct sentence from a (grammatical, ungrammatical) sentence pair.
  - Spoken StoryCloze [Hassid et al., 2024] evaluates semantic comprehension by assessing the model's capability to select the genuine ending of a story from a pair of ending choices.

- All the evaluation is conducted by comparing the model's negative log-likelihood of the data pair.

#### Paralinguistic Evaluation
In contrast to linguistic evaluation, paralinguistic evaluation focuses on the non-verbal aspects of communication that accompany speech.

- Some works choose to utilize paralinguistic tokens alongside semantic tokens to enhance the paralinguistic abilities of SpeechLMs [Kharitonov et al., 2021; Nguyen et al., 2024], so one way is to evaluate the paralinguistic tokens.

- pGSLM [Kharitonov et al., 2021] measures the correctness, consistency, and expressiveness of the prosodic tokens:
  - Correctness evaluates the model's ability to generate accurate prosodic profiles by calculating the minimal mean absolute error (min-MAE) of the prosodic tokens from 20 generated samples against the prosodic tokens from the reference.
  - Consistency is assessed through the Pearson correlation between the mean values of the prompt prosodic and its generated continuation prosodic tokens.
  - Expressiveness is measured by the standard deviation of the generated prosody token values, with the expectation that it matches the variability of the ground truth.

- [Nguyen et al., 2024] propose to measure on the perceptual level. They introduced a speech-text sentiment preservation benchmark (STSP), which requires the model to generate a text or speech sequence of tokens that preserves the sentiment of the prompt. A sentiment classifier is used to assess the sentiment in the generated speech. It should be noted that although they only apply the preservation approach on sentiment, this idea can be generalized to other paralinguistic features, such as timbre or prosody.

#### Generation Quality and Diversity
Quality and diversity are two crucial aspects of model generation. Typically, there is a trade-off between these dimensions when sampling model responses at different temperatures.

- GSLM [Lakhotia et al., 2021] suggests using the Area Under the Curve (AUC) with various temperature values. Specifically, AUC on perplexity and VERT are employed to assess these factors, where VERT represents the geometric mean of the ratio of k-grams in the generated speech that appears at least once.

- Additionally, the ChatGPT score can be utilized to evaluate the quality of the generated speech. In this process, the generated speech is transcribed using state-of-the-art ASR models and then sent to ChatGPT for quality (and diversity) assessment.

#### Downstream Evaluation
Refers to evaluating the ability of SpeechLMs to perform specific tasks, such as ASR, TTS, Speaker Identification, etc.

- The evaluation can be performed on pre-trained models by adding few-shot example(s) at the start of the prompt or on the instruction-tuned models by directly instructing them to do so.

- SUPERB [Yang et al., 2021] is a benchmark containing a wide range of downstream tasks that can be performed by SpeechLMs.

### Human (Subjective) Evaluation
This type of evaluation relies on human judgment to assess the quality of the outputs generated by SpeechLMs.

#### Mean Opinion Score (MOS)
Metric quantifies the perceived quality of speech output as judged by human listeners.

- Typically, a group of evaluators listens to a series of audio samples generated by the SpeechLM and rates each sample on a predefined scale, often from 1 (poor quality) to 5 (excellent quality).

- MOS is calculated by averaging the scores given by all evaluators for each audio sample, providing a single score that reflects the overall quality as perceived by humans.

- Variations of MOS focus on different aspects of speech quality, including MMOS, PMOS, and SMOS [Kharitonov et al., 2021; Zhang et al., 2024]. They evaluate the aspects of naturalness, prosody, and timbre similarity of the given speech, respectively.

- Typically, evaluating naturalness or timbre similarity involves collecting human opinions. However, this process can be complicated due to the challenges of recruiting participants and gathering their evaluations.

- As a result, researchers often turn to machine-based evaluations. They commonly employ neural network models specifically trained for these tasks. For instance, a naturalness prediction model [Mittag et al., 2021] can assess the naturalness of generated outputs, while a speaker identification model can evaluate timbre similarity.

## Challenges and Future Directions

### Understanding Different Component Choices
Current research on SpeechLMs encompasses key components such as speech tokenizers, language models, and vocoders, each offering a diverse range of options.

- There remains a significant gap in understanding the advantages and disadvantages of different component selections.

- Therefore, studies aimed at comprehensively comparing these choices are essential. Such an investigation would yield valuable insights and serve as a guide for selecting more efficient components when developing SpeechLMs.

### End-to-End Training
Although SpeechLMs can generate speech directly without relying on text signals, they still need to train the three components separately. This separate optimization may hinder the model's overall potential.

- Consequently, it would be worthwhile to investigate whether training can be conducted in an end-to-end manner, allowing gradients to be back-propagated from the vocoder's output to the tokenizer's input.

- By exploring this fully end-to-end approach, we could potentially enable SpeechLMs to produce more coherent, contextually relevant, and high-fidelity speech outputs.

### Real-Time Speech Generation
Enabling real-time speech generation is crucial in SpeechLM as it fosters a more interactive way of engaging with humans.

- However, the most adopted approaches described in section 3 still result in noticeable delays between input and output speech generation. This delay occurs because a typical vocoder must wait for the entire sequence of output tokens to be generated by the language model before functioning, making it the most time-consuming process in the inference pipeline.

- One potential solution to improve latency is to develop a streamable vocoder, allowing it to begin synthesizing output speech while the language model generates output speech tokens.

- Another option could involve the SpeechLM autonomously generating audio samples in waveform. Overall, this area of real-time speech generation remains underexplored and requires further investigation.

### Safety Risks in SpeechLMs
Primary concerns for the safety issues in SpeechLMs include but are not limited to toxicity and privacy.

- Toxicity refers to the harmful nature of the content generated by SpeechLMs.

- Privacy involves the risk of revealing personal information from the speech input after it has been processed by a SpeechLM.

- Even more concerning is the potential for the model to make biased inferences.

### Performance on Rare Languages
SpeechLMs directly model speech data, which allows them to more effectively handle "low-resource" languages compared to TextLMs.

- "Low-resource" languages are those that lack extensive textual data, making it challenging for TextLMs to model them efficiently. In contrast, SpeechLM provides a better solution by modeling the speech data of these "low-resource" languages, which often have more available audio data than text [Lakhotia et al., 2021].

- Therefore, future research could focus on training SpeechLMs in "low-resource" languages or dialects to expand their capabilities.

