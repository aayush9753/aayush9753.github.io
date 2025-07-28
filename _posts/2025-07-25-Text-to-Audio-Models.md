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
2. **Clean‑up:** remove missing audio → 5479 clips.
3. **Split:** TrainA / TestA.
4. **Control sentences:** append 0–4 beat/chord/key/tempo lines → TrainB / TestB.
5. **Paraphrase:** ChatGPT rephrasing → TrainC (85 % paraphrased).
6. **Filter:** drop “poor‑quality/low‑fidelity” captions → 3413 HQ clips.
7. **11× augment:** ±1‑3 semitones, ±5–25 % speed, crescendo/decrescendo volume → ≈37 k new samples.
8. **Merge:** TrainA + B + C + augments → **52 768** training items; tests = TestA, TestB, FMACaps.

**Mustango Model**
1. **Latent space:** AudioLDM VAE → latent z.
2. **Diffusion:** 200‑step Gaussian forward / reverse process.
3. **MuNet denoiser:** UNet + hierarchical cross‑attention.
   * Inputs: FLAN‑T5 text emb; beat & chord encodings.
4. **Beat encoder:** one‑hot type + time sinusoid.
5. **Chord encoder:** root FME + quality + inversion + time sinusoid.
6. **Training tricks:** 5% unconditional, 5% single‑modality drop, 20‑50% sentence masking; AdamW, batch 32.
7. **Inference helpers:**
   * DeBERTa beat predictor (meter + intervals).
   * FLAN‑T5 chord predictor (time‑stamped chords).
   * Classifier‑free guidance scale = 3.
8. **Output:** 10‑s waveform obeying tempo, key, chords, beats when provided; graceful fallback when not.


### 2. [Noise2Music: Text-conditioned Music Generation with Diffusion Models](https://aayush9753.github.io/noise2music-text-conditioned-music-generation-with-diffusion-models.html)
Generate a 30-second, 24 kHz stereo music clip from a plain-language prompt.
### Noise2Music — Key Take‑aways in Bullet Form

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

---

*More blog posts coming soon as I continue my learning journey...*
