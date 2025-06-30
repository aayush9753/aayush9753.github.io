---
layout: page
title: "Recent Advances in Speech Language Models: A Survey"
permalink: /dm/speech-language-models-survey/
---

# Sesame: Conversational speech generation - TTS

[Sesame: Crossing the Uncanny Valley of Voice (Demo)](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo)

## Model Architecture

**Two-component design:**

- **Multimodal Backbone Transformer:** Processes interleaved text and audio, models zeroth codebook
- **Audio Decoder:** Smaller transformer that models remaining N-1 codebooks for speech reconstruction
- Both transformers are variants of the Llama architecture

**Tokenization:**
- Text: Llama tokenizer
- Audio: Mimi split-RVQ tokenizer (one semantic + N-1 acoustic codebooks per frame at 12.5 Hz)

**Three model sizes:**
- Tiny: 1B backbone, 100M decoder
- Small: 3B backbone, 250M decoder
- Medium: 8B backbone, 300M decoder

**CSM model inference process:**
- Text (T) and audio (A) tokens are interleaved and fed sequentially into the Backbone, which predicts the zeroth level of the codebook.
- The Decoder then samples levels 1 through N–1 conditioned on the predicted zeroth level.
- The reconstructed audio token (A) is then autoregressively fed back into the Backbone for the next step, continuing until the audio EOT symbol is emitted.
- This process begins again on the next inference request, with the interim audio (such as a user utterance) being represented by interleaved audio and text transcription tokens.

## Training Data

- Large-scale dataset of publicly available audio
- ~1 million hours of predominantly English audio
- Processing pipeline: transcription → diarization → segmentation → filtering

## Training Process

- Sequence length: 2048 (approximately 2 minutes of audio)
- Training duration: Five epochs
- **Compute Amortization:** Novel approach to reduce memory burden
  - Backbone: Processes zeroth codebook for all frames
  - Decoder: Trained on only random 1/16 subset of audio frames
  - No perceivable difference in audio decoder losses observed

**Amortized training process:**
- The backbone transformer models the zeroth level across all frames (highlighted in blue), while the decoder predicts the remaining N–1 levels, but only for a random 1/16th of the frames (highlighted in green).
- The top section highlights the specific frames modeled by the decoder for which it receives loss.

## Novel Contributions

### Architectural Innovations
- Single-stage model: Improves efficiency and expressivity compared to two-stage approaches
- Split transformer design: Separates backbone and decoder functions at zeroth codebook
- Compute amortization scheme: Successfully validated that training on a subset of frames (1/16) maintains quality while significantly reducing memory requirements

### Evaluation Innovations
- Developed new evaluation suite addressing context-aware speech capabilities
- New benchmarks:
  - **Homograph Disambiguation:** Validated improved pronunciation accuracy with model scaling (tested with 5 homographs: lead, bass, tear, wound, row)
  - **Pronunciation Continuation Consistency:** Confirmed that larger models maintain better consistency (tested with 10 words having common variants: aunt, data, envelope, mobile, route, vase, either, adult, often, caramel)
  - **Context-dependent CMOS studies:** Proved that providing conversational context affects human perception of appropriate speech

## Key Findings

- Performance consistently improved with model scaling, validating the hypothesis that larger models produce more realistic speech
- Successfully demonstrated the importance of context in speech generation through subjective studies
- Identified that the model acquires some multilingual capability through "dataset contamination" despite English-focused training

## Inference in Real-World Use Cases

**The inference process follows this sequence:**
1. Text (T) and audio (A) tokens are interleaved and fed into the Backbone Transformer
2. Backbone predicts the zeroth level codebook for each frame
3. Decoder samples levels 1 through N-1 conditioned on the predicted zeroth level
4. Reconstructed audio token is autoregressively fed back into the Backbone
5. Process continues until an end-of-token symbol is emitted
6. Cycle repeats with each new inference request

**In a real conversational setting:**
- System maintains history of conversation (both text and audio)
- User utterances are represented by interleaved audio and text transcription tokens
- Model produces contextually appropriate responses considering previous exchanges
- Speed is optimized by the smaller decoder component and compute amortization

## Open-Source Components

- The researchers are open-sourcing:
  - The models under Apache 2.0 license
  - Access via GitHub repository (mentioned but specific components not detailed)
- The open-source release appears to focus on the models themselves rather than the training data or full codebase, though the exact scope is not explicitly defined in the paper.

## Limitations and Future Roadmap

**Current limitations:**
- Primarily English-only
- Doesn't leverage pre-trained LM weights

**Planned improvements:**
- Scale up model size
- Increase dataset volume
- Expand language support to 20+ languages
- Explore integration with pre-trained language models
- Develop fully duplex models that better handle conversation dynamics