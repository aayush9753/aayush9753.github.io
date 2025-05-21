---
layout: page
title: FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities 
in Speech Dialogue Systems
permalink: /dm/flexduo/
---

# FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities in Speech Dialogue Systems

Liao, Borui, Yulong Xu, Jiao Ou, Kaiyuan Yang, Weihua Jian, Pengfei Wan, and Di Zhang. "FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities in Speech Dialogue Systems." arXiv, February 19, 2025. [https://doi.org/10.48550/arXiv.2502.13472](https://doi.org/10.48550/arXiv.2502.13472).

## Model Architecture

FlexDuo uses Qwen2-audio-7B-Instruct as its base model with the following configuration:

- **Audio Encoder**: Kept frozen to maintain audio processing capabilities
- **LLM component**: Fine-tuned for duplex interaction alignment
- **Special tokens**: Added to the LLM vocabulary for dialogue actions

![FlexDuo Model Architecture]({{ site.baseurl }}/assets/images/dm/flexduo_architecture.png)

The overall architecture consists of three main components:

### State Manager

Predicts dialogue actions using a finite state machine. Follows a finite state machine with 7 possible transition actions:

- **K.S (Keep Speaking)**: The assistant maintains the current Speak state.
- **K.L (Keep Listening)**: The user has not finished speaking.
- **K.I (Keep Idling)**: Environmental noise, user's backchannels, or third-party dialogue.
- **S2L (Speak to Listen)**: The user interrupts the assistant.
- **S2I (Speak to Idle)**: The assistant finishes speaking and ends naturally.
- **L2S (Listen to Speak)**: The assistant responds to or interrupts the user.
- **I2L (Idle to Listen)**: The user starts speaking.

![State Transition Diagram]({{ site.baseurl }}/assets/images/dm/flexduo_state_transitions.png)

### Context Manager
Maintains dialogue states and context history

### Sliding Window
Monitors real-time audio input (size set to 5 in experiments)

Our duplex control module predicts the current dialogue strategy based on the context of historical dialogues, the current state, and the accumulated speech chunks in the sliding window.

## Data Processing Pipeline

### Dataset Source:

- Fisher corpus (English: 831 hours, Chinese: 389 hours)
- Contains separate recordings for two speakers in conversations

![Data Processing Pipeline]({{ site.baseurl }}/assets/images/dm/flexduo_data_pipeline.png)

### Data Preprocessing:

- Used VAD to extract Inter-Pausal Units (IPUs)
- Merged IPUs with gaps less than 160ms
- Labeled completely overlapped IPUs as backchannels
- Others labeled as valid dialogue turns

### Quality Control:

- Transcribed dialogues using ASR
- Filtered using GPT-o1-mini dialogue evaluation
- Human validation showed 91.6% acceptance rate
- Final dataset: 671 hours English, 263 hours Chinese

### Training Data Organization:

- Applied sliding window to re-segment audio streams
- Labeled based on VAD information (turn transitions, utterance status, system state)
- Incorporated human preference in labeling process
- Added 500ms delay for "listening phase" to distinguish backchannels from interruptions

## Training Process

### Configuration:

- Standard cross-entropy as training objective
- Loss masking applied to historical context and user audio input
- AdamW optimizer with learning rate 1e-5
- Batch size of 3 per GPU
- 40,000 training steps with 500-step warm-up
- Hardware: 8 NVIDIA H800 GPUs
- DeepSpeed ZeRO-3 optimization

### Dataset Split:

- 10:1:1 ratio for training, validation, and test sets

FlexDuo defines the Assistant's turn-taking as occurring strictly after the User's turn ends. However, the timing for the User's turn-taking remains unchanged to fully simulate real-world scenarios where users interrupt at specific moments.

![Training Process]({{ site.baseurl }}/assets/images/dm/flexduo_training.png)

## Novel Contributions

### Pluggable Architecture:

- Decoupled design separates duplex control from dialogue system
- Proven effective through integration with GLM4-voice and MiniCPM-o
- Demonstrated 23.1% reduction in false interruption rate vs. VAD-controlled systems

### Idle State Introduction:

- Three-state model (Speak, Listen, Idle) vs. traditional two-state approach
- Ablation studies validated this by showing:
  - 15.68% decrease in Turn-taking F1 score when Idle state removed
  - 13.62% increase in false interruption rate when Idle state removed

### Sliding Window Optimization:

- Controlled experiments with window sizes w ∈ {2, 4, 8}
- Demonstrated trade-off between responsiveness and semantic understanding
- Selected optimal size to balance these factors

### Semantic Integrity Buffer:

- 500ms delay mechanism for listening phase
- Provides time for semantic judgment before state transitions
- Empirically shown to reduce contextual noise interference

![Novel Contributions]({{ site.baseurl }}/assets/images/dm/flexduo_contributions.png)

## Real-World Inference Process

When deployed, FlexDuo operates as follows:

### Initial Setup:

- Integrates with an existing half-duplex LLM (e.g., GLM4-voice)
- Initializes with Idle state

### Real-time Processing:

- Every 120ms, evaluates audio and context
- Manages the sliding window based on current state
- Predicts one of seven dialogue actions using the state manager

### Dialogue Flow Control:

- Filters audio in Idle state to maintain clean context
- Passes relevant audio to the half-duplex LLM when in Listen state
- Controls when the LLM generates responses (Speak state)
- Interrupts LLM output when state changes to Idle or Listen

![Inference Process]({{ site.baseurl }}/assets/images/dm/flexduo_inference.png)

### Context Management:

- Updates dialogue context when state transitions occur
- Maintains semantic integrity of the conversation
- Example: user backchannels (like "hmm") in the middle of a conversation are filtered out when in Idle state

### Response Generation:

- Half-duplex LLM generates responses based on filtered context
- System can transition states mid-utterance if needed
- Avoids false interruptions due to background noise or backchannels

## Performance Validation

### Interaction Capability:

- 24.9% reduction in false interruption rate vs. integrated full-duplex systems
- 7.6% improvement in turn-taking performance vs. integrated systems

### Dialogue Quality:

- 35.3% reduction in conditional perplexity on English Fisher dataset
- 19% reduction in conditional perplexity on Chinese Fisher dataset
- Demonstrates that context filtering meaningfully improves dialogue coherence

### Human Evaluation:

- 91.6% acceptance rate on the dialogue quality filtering process
- Validates the effectiveness of the data preparation approach

![Performance Results]({{ site.baseurl }}/assets/images/dm/flexduo_performance.png)

The system's real-world performance demonstrates that FlexDuo successfully addresses the limitations of existing full-duplex systems while maintaining compatibility with half-duplex LLMs, providing a more natural conversational experience without the need for complete system retraining. 