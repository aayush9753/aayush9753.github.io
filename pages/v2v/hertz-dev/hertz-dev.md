[Blog] [Code] [Models]

Inference code and model checkpoints provided

Simple model architecture

Training code not provided

Data info is not provided

Contributions

8.5B, full-duplex, audio-only base model: hertz-dev.

120ms real-world latency on a single RTX 4090.

## Model
Made up of hertz-codec and hertz-ar. The hertz-codec produces audio latents and the hertz-ar predicts future latents conditioned on past latents. The audio latents are an extremely rich prior that could be used for many downstream tasks.

### hertz-codec: convolutional audio VAE
- **Input:** Mono, 16kHz
- **Output:** Encodes a 8Hz latent representation with a KL-regularized 1kbps bitrate.
- 32-dim latent per 125ms frame
- ![hertz-codec architecture](image-20241224-153016.png)
- **VAE and KL regularization:** Utilize causal convolutions (functionally adding padding to the left of the sequence) to enable streaming inference.
- Outputs gaussian parameters (means and variances) that are sampled into just a single 32-dim latent per 125ms frame.
- Hertz-codec has 5 million encoder parameters and 95 million decoder parameters.

#### Weights:
- `inference_apatosaurus_95000.pt` — hertz-codec weights trained on a mixed reconstruction, adversarial, and KL-regularized loss.
- `inference_volcano_3.pt` — hertz-codec quantizer, a learned projection distilling the most phonetically relevant 15-bits of each latent.

### hertz-ar
- 8.4B decoder-only with a context of 2048 input tokens (~4.5 mins).
- The model uses a 40-layer transformer divided into two main sections:
  - The first 32 layers focus on understanding the audio context and creating a compressed representation
  - Takes the rich 32-dimensional latent representations and compresses it into 15 discrete bits.
  - The last 8 layers then utilize the latent history and the 15 bit quantized latent to predict future latent audio tokens.
  - First using the quantized 15-bit representation to get a rough idea of the next token
  - Then using the full context (both history and this rough prediction) to generate a more refined prediction
  - The output is a latent that can be passed into hertz-codec.
- hertz-lm can be trained independently or initialized from language model weights.
- Duplex audio is handled as a post-training task with two projection heads concatenated together, then separated into two quantized projection pipelines conditioned on their respective residuals.
- ![hertz-ar architecture](image-20241224-153025.png)

#### Weights:
- `inference_caraway_112000.pt` — hertz-lm weights initialized from a language model trained on 2T tokens.
- `inference_syrup_110000.pt` — hertz-lm weights initialized randomly and fully trained on audio latents.
- `inference_whip_72000.pt` — hertz-ar weights for the last 8 layers
- `inference_care_50000.pt` & `inference_scion_54000.pt` — Duplex checkpoints for hertz-ar

## Results
Base models accurately predict the distribution of the data that they were trained on, as opposed to models that have had substantial RL tuning done to collapse their generation distributions.

This makes these models the best starting point for downstream fine-tuning in a large number of different tasks.

Currently training a larger, more advanced version of Hertz, which will use a scaled base model recipe and RL tuning to substantially improve the raw capabilities and final coherence of the model.

## Training Choices
- Causal ConvNets were used in hertz-codec for parallel decoding + more granular control over latent generation.
  - **Parallel Decoding:** Causal ConvNets can process multiple timesteps in parallel
  - **Granular Control Over Latent Generation:** Maintain temporal dependencies (each output depends only on past inputs) which can be controlled by the receptive field.

### hertz-ar
- 15bit quantized latents were initially trained to contain phonetics which helps steer the model to create syntactically correct speech.
- Quantization was done via an MLP projection into a Finite Scalar Quantization layer.
- Model recipe effectively learns linguistics both with and without text model initialization.

## Performance
- **Live Inference:** 8 forward passes per second
- **Dual Channel:** It takes two separate channels as input, but in conversations only returns one. At each step, it receives the human's audio and tokenizes it into a latent, combining this with the model's last generated latent and feeding both into hertz-ar.
- **Latency:** Measured as the average time between user utterance and model response, to be:
  - 62.5ms (the average time between any given utterance and the end of one token) + the time for forward pass + round-trip internet delay.
  - 120 ms (local 4090s)
