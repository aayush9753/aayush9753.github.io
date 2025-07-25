---
layout: post
title: Text-to-Audio-Models
category: journey
date: 2025-07-25
---

## My Journey into Text to Audio Models

I am studying text to audio models with more focus towards Music Generation models.

# Text-To-Music Models

### 1. [Mustango: Toward Controllable Text-to-Music Generation](https://aayush9753.github.io/mustango-toward-controllable-text-to-music-generation.html)
Mustango is a diffusion-based text-to-music model that enables structured control over **chords, beats, tempo,** and **key** directly from natural-language prompts.
It covers:
- Fundamentals of Music
- MuNet: Music-Aware UNet Denoiser
- MusicBench: A Richly Augmented Dataset - built on top of MusicCaps. (Over 52000 samples)

### 2. [Noise2Music: Text-conditioned Music Generation with Diffusion Models](https://aayush9753.github.io/noise2music-text-conditioned-music-generation-with-diffusion-models.html)
- **Goal:** Generate a 30-second, 24 kHz stereo music clip from a plain-language prompt.
- **Approach:** Uses a cascade of several diffusion models:
  1. **Generator:** Produces a low-resolution latent audio from text.
  2. **Cascader(s):** Upsample and refine the latent to a 16kHz waveform.
  3. **Super-resolution:** Final stage upsamples to 24kHz.
  - All models are 1D U-Nets.
- **Intermediate representations:** Either log-mel spectrogram or low-fidelity (3.2kHz) audio.
- **Results:** Generated audio matches prompt details (genre, tempo, instruments, mood, era) and captures fine semantics.
- **Data:** 
  - Text labels are auto-generated:
    1. LLM creates many descriptive caption candidates.
    2. A music-text embedding model scores and selects the best captions for each audio clip.
    3. ~150,000 hours of audio are annotated this way.


---

*More blog posts coming soon as I continue my learning journey...*
