---
layout: post
title: Audio Signal Processing for ML [WIP]
category: [speech, ongoing]
date: 2025-04-28
---

# Introduction

This post explores fundamental concepts in audio signal processing that form the backbone of machine learning applications for sound and speech. The content is based on the [Audio Signal Processing for Machine Learning](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0) video series, with code examples available in the [author's GitHub repository](https://github.com/musikalkemist/AudioSignalProcessingForML/tree/master).

## Prerequisites

For understanding Fourier analysis concepts:
- [Fourier series introduction](https://www.youtube.com/watch?v=UKHBWzoOKsY)
- [Fourier Transform](https://www.youtube.com/watch?v=spUNpyF58BY) - Decomposing complex sounds into their constituent frequencies

# Sound Fundamentals

## Sound and Waveforms
- **Sound**: Produced by vibration of an object. Vibrations cause air molecules to oscillate and changes in air pressure create waves.
- **Key Properties**:
  - Mechanical Wave
  - Frequency (Hz)
  - Amplitude
  - Phase
- Higher frequency produces higher-pitched sound, while higher amplitude results in louder sound
- **Pitch**: Perception of frequency. One octave has 12 notes, and frequency doubles after one octave.

## Intensity and Loudness
- **Sound Power**: Measured in watts (W). Energy per unit of time by sound source in all directions.
- **Sound Intensity**: Sound power per unit area, measured in decibels (dB)
  - Threshold of Hearing (TOH): 10^(-12) W/m^2
  - Threshold of Pain (TOP): 10 W/m^2
- **Decibel Formula**: dB(I) = 10 log(I/I_toh) [log base 10]
- **Common Reference Points**:
  - Normal conversation: 60dB
  - Jet take-off: 140dB
- **Loudness (phons)**: Subjective perception of sound intensity, dependent on duration and frequency of sound

## Timbre
- The quality that distinguishes sounds of same intensity, frequency and duration
- **Key Characteristics**:
  - Multidimensional in nature
  - **Sound envelope**: Attack-Decay-Sustain-Release model
  - **Harmonic content**:
    - Superposition of sinusoids
    - A partial is a sinusoid used to describe sound
    - The lowest partial is called fundamental frequency
    - A harmonic partial is a frequency that's a multiple of fundamental frequency
    - Inharmonicity indicates a deviation from a harmonic partial
  - **Modulation**:
    - Amplitude modulation → Tremolo
    - Frequency modulation → Vibrato

# Digital Audio Processing

## Analog to Digital Conversion
- **Analog Signal**: Continuous value of time and amplitude
- **Digital Signal**: Sequence of discrete values
- **Pulse Code Modulation** process:
  - **Sampling**:
    - Sampling rate (sr): Number of samples per second
    - Nyquist frequency: sr/2 - Upper bound for frequency in digital signal that won't create artifacts
    - Potential artifacts: Aliasing
  - **Quantization**:
    - Resolution = number of bits (same as bit depth)
    - Dynamic range: Difference between largest and smallest signal a system can record
    - Signal-to-quantization-noise ratio (SQNR): Relationship between maximum signal strength and quantization error

## MEL Spectrograms and Audio Features for Machine Learning
Audio features provide meaningful representations of sound for machine learning algorithms to process.