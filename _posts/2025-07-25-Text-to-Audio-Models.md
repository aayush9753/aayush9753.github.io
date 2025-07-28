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

**MusicBench — Dataset Pipeline**
1. **Seed corpus:** 5521 MusicCaps clips (10 s + captions).
2. **Control sentences:** append 0–4 beat/chord/key/tempo lines
3. **Paraphrase:** ChatGPT rephrasing
4. **Filter:** drop “poor‑quality/low‑fidelity” captions
5. **11× augment:** ±1‑3 semitones, ±5–25 % speed, crescendo/decrescendo volume → ≈37 k new samples.

**Mustango Model**
1. **Latent space:** AudioLDM VAE → latent z.
2. **MuNet denoiser:** UNet + hierarchical cross‑attention.
   * Inputs: FLAN‑T5 text emb; beat & chord encodings. (**Beat encoder** and **Chord encoder**)
3. **Inference helpers:**
   * DeBERTa beat predictor (meter + intervals).
   * FLAN‑T5 chord predictor (time‑stamped chords).
4. **Output:** 10‑s waveform obeying tempo, key, chords, beats when provided; graceful fallback when not.


### 2. [Noise2Music: Text-conditioned Music Generation with Diffusion Models](https://aayush9753.github.io/noise2music-text-conditioned-music-generation-with-diffusion-models.html)
Generate a 30-second, 24 kHz stereo music clip from a plain-language prompt.

**Training‑Data Pipeline**
1. **Raw audio pool:** 6.8 M full‑length tracks → chopped into 30 s clips (\~340 k h).
2. **Caption vocabularies (built offline)**
   * **LaMDA‑LF** – 4M rich sentences (LLM‑generated).
   * **Rater‑LF / SF** – 35k long + 24k short human sentences/tags from MusicCaps.
3. **Embedding space scoring:** Encode every clip (MuLan‑audio) & every caption (MuLan‑text).
4. **Pseudo‑labelling:** For each clip pick top‑10 captions by cosine sim → sample 3 low‑frequency ones from each vocab (bias toward rarer labels).
5. **Extra metadata:** Append title, artist, genre, year, instrument tags.
6. **Quality anchor:** Inject \~300 h curated, attribution‑free tracks with rich manual metadata.
7. **Dual‑rate storage:** Keep 24 kHz (for super‑res stage) + 16 kHz copies (for the rest).
8. **Final payload:** Every 30 s clip carries 10 + text descriptors spanning objective tags → subjective vibes.

**Model Stack (three‑stage diffusion cascade)**

| Stage                  | I/O                             | Role                     | Key details                                                                         |
| ---------------------- | ------------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| **Waveform Generator** | *Text → 3.2 kHz audio*          | Sketch global structure. | 1‑D Efficient‑U‑Net; text fed via cross‑attention; CFG during sampling.             |
| **Waveform Cascader**  | *Text + 3.2 kHz → 16 kHz audio* | Upsample & refine.       | Receives up‑sampled low‑fi audio + prompt; blur/noise augmentation during training. |
| **Super‑Res Cascader** | *16 kHz → 24 kHz audio*         | Restore full bandwidth.  | No text conditioning; lightweight U‑Net.                                            |

> **Spectrogram path** (alt): parallel generator + vocoder pair that works in log‑mel space; cheaper but less interpretable.

### 3. [Stable Audio - Fast Timing-Conditioned Latent Audio Diffusion](https://aayush9753.github.io/stable-audio.html)
1. A **convolutional VAE** that efficiently compresses and reconstructs long stereo audio.
2. It uses **latent diffusion**
3. It adds **timing embeddings**.

**Dataset Construction**
1. **Collect** 806284 stereo tracks (≈ 19500 h) from **AudioSparx**.
2. **Pre‑process audio**
   * Resample to **44.1 kHz**, stereo.
   * Slice / pad each file to a fixed **95.1 s** window (4 194 304 samples).
3. **Build text prompts** from metadata *on‑the‑fly*
   * Randomly sample descriptors (genre, mood, BPM, instruments).
   * Emit either **free‑form** or **structured** text strings.
4. **Final sets**
   * Same corpus trains the **VAE**, **CLAP** (text encoder), and **latent diffusion U‑Net**.

**Model Pipeline**

| Stage                          | Key points                                                                                                                        |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **1. VAE**     | 32× compression                                  |
| **2. Text encoder (CLAPours)** | trained from scratch                             |
| **3. Timing embeddings**       | seconds_start, seconds_total; concatenated with text features |
| **4. Latent U‑Net diffusion**  | 907 M params;                  |
| **5. Inference**               | DPMSolver++              |

**Outcome:** 44.1 kHz stereo audio, up to 95 s, fast (latent) diffusion with precise duration control via timing conditioning.

---

*More blog posts coming soon as I continue my learning journey...*
