---
layout: page
title: "FlexDuo - A Pluggable System for Enabling Full-Duplex Capabilities in Speech Dialogue Systems"
permalink: /dm/flexduo/
---

[Paper Link](https://arxiv.org/abs/2412.04917)

## 3. Flow-Omni

### 3.1 Architecture

![Flow-Omni Model Architecture]({{ site.baseurl }}/assets/images/dm/flow-omni_architecture.png)

#### Speech input module

Flow-Omni employs the Whisper encoder as its speech encoder.

The Whisper model features a highly robust architecture, comprising an audio encoder designed to transform raw audio waveforms into compact, high-dimensional continuous speech embeddings. 

This architecture is particularly adept at handling speech inputs across diverse dimensions, including various languages, accents, prosodies, and background noise levels. 

Additionally, Whisper's pre training on an extensive and diverse dataset endows it with strong generalization capabilities, enabling it to effectively capture and encode speech features, even in low-resource or unseen conditions. 

Following the whisper encoder, several linear blocks are used to further project the speech features to LLM hidden dimensions acting as speech input tokens for LLM training. 

#### Pretrained language model

Flow-Omni integrates Qwen2 as the LLM decoder for Chinese dataset experiments. 

Its inherent compatibility with multimodal frameworks facilitates seamless integration with the Whisper encoder. 

The pretrained model weight is first frozen for modality alignment then all the model parameters are finetuned to further improve generation and understanding quality.

#### Speech output module

##### Issues with discrete token approach:

The discrete token approach involves converting speech signals into a sequence of quantized ids, typically achieved via vector quantization or clustering methods. 

However, quantization introduces information loss, and the discrete nature of tokens can make it challenging to capture subtle variations in speech, like locale, prosody and style. 

Additionally, discrete representations may struggle with domain transfer, as the token vocabulary is tied to the training data distribution and may fail to generalize to unseen acoustic conditions or languages. 

Besides, the discrete token approach relies heavily on a high-quality pretrained codec as well as its deliberated design on codebook size and codebook number, etc. 

##### Continuous token method

Continuous token method bypasses the quantization step and directly predicts detailed acoustic features. 

This approach allows for the generation of highly realistic speech, as the model retains access to the full spectrum of acoustic information. 

Moreover, continuous token models are not limited by a fixed vocabulary, providing more flexibility, and can be seamlessly integrated with neural vocoders to further improve speech quality. 

Speech output module is for predicting continuous-valued mel-spectrogram in Flow-Omni. 

##### Mel-spectrogram as the speech representation

The model generates speech in the form of mel-spectrogram frames, which are 2D representations of audio signals (time vs. frequency).

##### Conditional probability density paths (vector field approach)

Instead of directly predicting the mel-spectrogram in one shot, the system models a continuous transformation from random noise into the final mel-spectrogram.

This transformation is guided by a conditional vector field—basically, a learned function (via a small MLP) that says "move from your current noisy state toward the true mel-spectrogram" at each step.

##### Transformer-based conditioning

A transformer processes the output of the language model (the "hidden state") to capture acoustic-relevant information—like what words, phonemes, or intonations should come next.

That hidden state is fed into the small MLP (the vector field model) to steer the denoising process, ensuring the generated mel-spectrogram aligns with the intended text or acoustic context.

##### Progressive denoising

Starting from random noise, the model applies the conditional vector field multiple times (each time guided by the MLP and transformer outputs) to gradually "clean up" the noise into a valid mel-spectrogram frame.

##### GAN-based vocoder

Once the mel-spectrogram is predicted, it's passed to a vocoder (a generative adversarial network) to convert the spectrogram into a final waveform—i.e., an actual audio signal that can be played back.

##### Autoregressive prediction of the next frame

The newly predicted mel-spectrogram frame is also fed back into the language model decoder (or a similar mechanism) so that the model knows how it just sounded.

This feedback loop lets the model predict the next mel-spectrogram frame, continuing the audio generation sequence step by step.

In short, the system uses:

- A transformer (conditioned on language model outputs) to encode what needs to be generated.
- A conditional vector field (small MLP) to iteratively "denoise" from random noise to a mel-spectrogram.
- A GAN vocoder to convert the mel-spectrogram into real audio.
- An autoregressive loop so that each new frame depends on the previously generated frames.

### 3.2 Parallel Sequence Modeling

Flow-Omni extends speech capabilities in language models by jointly modeling a parallel speech sequence and text sequence. 

Unlike approaches that maintain multiple speech sequences, Flow-Omni uses one speech state sequence (e.g., "start," "pad," "stop," "generating") while predicting mel-spectrograms through a dedicated speech output module. This design enables the model to accept either speech or text input and produce either speech or text output.

Four task types are covered, each with its own token sequence formulation (as shown in Figure 3). 

During input, the text or speech sequence is padded when not in use, ensuring parallel processing of both modalities. 

If the response is speech, the text tokens are generated in parallel but with a slight delay for the first speech frame—this leverages the text context before speech frames are produced. 

For speech inference, the model only predicts mel-spectrogram frames when the speech state is "generating," appending them to the autoregressive sequence in real time. This approach improves end-to-end speech–text interaction by effectively synchronizing textual and auditory information.

### 3.3 Training Objective

### 3.4 Inference Sampling

## 4. Experiment

### 4.1 Datasets

In our experiments, we used the Chinese speech-text dataset Aishell-1, WenetSpeech and the base version of dialogue corpus LCCC.

**Aishell-1 and WenetSpeech**: The total recording duration of the Aishell-1 dataset is 178 hours, recorded by 400 different speakers. This dataset contains a large number of recordings covering a variety of accents, speech speeds, and emotional variations. The WenetSpeech dataset contains 10,000+ hours of Mandarin speech data, all from YouTube and Podcast, covering a variety of fields and speaking styles, including audiobooks, live commentaries, documentaries, dramas, interviews, news, reading, discussion, variety shows and more. These data cover different Internet audio and video, noise background conditions. We trained the model with the transcripts and the mel-spectrograms extracted from the original audios.

**LCCC-base**: The LCCC-base dataset contains 6.8 million conversation rounds, derived from Weibo conversational data and open-source conversation datasets, and has undergone a rigorous cleaning process. We first removed repeated punctuations from the text and filter out Q&A pairs less than five words in length and then use Azure TTS service to synthesize the text in the data set into Q&A speech. The question speech was synthesized into the voice of multiple speakers, and the answer speech was fixed into the voice of the same person, to obtain the Q&A dataset with both text and speech. Then the mel-spectrograms were extracted from the audios for the training of the model.

### 4.2 Training Details

We adopted a two-stage training strategy to make Flow-Omni extend the speech capability based on the pretrained language model.

In the first stage, we focused on modal alignment to ensure that the speech input and output modules can effectively extract audio features interpretable by the language model and synthesize speech outputs based on the language model. To achieve this, the parameters of the pretrained language model and the Whisper encoder were frozen, while the parameters of all other modules remained trainable. Training was conducted using the Aishell-1 and WenetSpeech datasets, with automatic speech recognition (ASR) and text-to-speech (TTS) serving as the training tasks. For the ASR task, the parallel sequence format aligned with the "speech-in text-output" depicted in figure 3. Similarly, for the TTS task, the parallel sequence format followed the output section of "text-in speech-output" in figure 3.

In the second stage, we fine-tuned the entire model, allowing all parameters to be trainable. This stage leveraged the LCCC-base dataset supplemented with speech data and incorporates all four tasks outlined in figure 3. The sequence formats were consistent with those specified in the figure. After this comprehensive training process, the resulting model can effectively understand and respond to both speech and text inputs, achieving seamless multimodal functionality. During the training process, we utilized the adam optimizer in conjunction with a cosine annealing learning rate scheduler to ensure efficient convergence. The initial learning rate for both training stages was set to 1e-4, gradually reduced to a minimum of 5e-6 according to the annealing schedule. All experiments were conducted using NVIDIA V100 GPUs.

