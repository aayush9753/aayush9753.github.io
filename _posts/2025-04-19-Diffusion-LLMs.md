---
layout: post
title: Diffusion Large Language Models
category: [llm, diffusion]
date: 2025-04-19
---

## Section 1: Fundamentals of Diffusion Models

### 1.1 Core Concept: Iterative Noising and Denoising

Diffusion probabilistic models, commonly referred to as diffusion models, represent a powerful class of generative models that have demonstrated state-of-the-art performance in various domains, particularly in generating high-fidelity images and audio.1 The core idea, inspired by principles from non-equilibrium thermodynamics 5, is to systematically destroy structure in data through an iterative forward diffusion process and then learn a reverse process that restores the structure, effectively generating data samples from noise.1

This generative paradigm contrasts significantly with other established methods like Generative Adversarial Networks (GANs) or standard Variational Autoencoders (VAEs).1 While GANs involve a complex adversarial training dynamic and VAEs rely on often simple surrogate posterior distributions, diffusion models employ a conceptually straightforward two-phase mechanism. The first phase, the forward process (also called the diffusion or noising process), is typically fixed and involves gradually adding noise (usually Gaussian) to an input data sample over a sequence of time steps, progressively corrupting the data until it resembles pure noise.1 The second phase, the reverse process (also known as the denoising or generative process), is learned. It aims to reverse the noising steps, starting from a sample of pure noise and iteratively refining it to produce a sample that resembles the original data distribution.1 This iterative refinement allows the model to potentially correct errors made in earlier steps and gradually build complex data structures.3

The thermodynamic analogy posits the initial data distribution as being far from equilibrium. The forward process models the diffusion towards a simple equilibrium distribution (e.g., Gaussian noise). The learned reverse process then simulates the reversal of this diffusion, guiding samples from the simple equilibrium state back towards the complex, structured data manifold.8

### 1.2 The Forward Process (Diffusion): Mathematical Formulation

The forward diffusion process is mathematically defined as a Markov chain.3 Starting with an initial data sample x0​ drawn from the true data distribution q(x0​), the process generates a sequence of latent variables x1​,x2​,…,xT​ by adding Gaussian noise at each discrete time step t, where t ranges from 1 to T. The transition probability at each step is defined by a conditional Gaussian distribution 3:

q(xt​∣xt−1​)=N(xt​;1−βt​​xt−1​,βt​I)

Here, xt​ represents the state (noisy data sample) at time step t, N(⋅;μ,Σ) denotes a Gaussian distribution with mean μ and covariance Σ, I is the identity matrix, and {βt​}t=1T​ is a pre-defined variance schedule.3 The values βt​ are typically small positive numbers (0<βt​<1) that control the amount of noise added at each step. The variance schedule is a crucial design choice; common schedules include linear increases in βt​ (e.g., from 10−4 to 0.02 1) or cosine schedules.3 These schedules ensure that the data is gradually corrupted. While typically fixed, the noise schedule itself can potentially be learned.1 The Markov property implies that the state xt​ depends only on the immediately preceding state xt−1​.3 The joint probability of the entire sequence of noisy samples given the initial data is the product of these transitions:

q(x1:T​∣x0​)=t=1∏T​q(xt​∣xt−1​) 3

A significant property of this Gaussian forward process is that we can sample xt​ at any arbitrary time step t directly from the initial data x0​ in closed form, without iterating through all intermediate steps.3 By defining αt​=1−βt​ and αˉt​=∏i=1t​αi​ 5, the conditional distribution q(xt​∣x0​) is also Gaussian:

q(xt​∣x0​)=N(xt​;αˉt​​x0​,(1−αˉt​)I) 3

This allows for efficient training, as we can directly compute a noisy version xt​ for any t using the reparameterization trick: sample a standard Gaussian noise vector ϵ∼N(0,I) and compute:

xt​=αˉt​​x0​+1−αˉt​​ϵ 3

As the number of steps T becomes large, αˉT​=∏t=1T​(1−βt​) approaches zero. Consequently, the distribution of xT​ converges to an isotropic Gaussian distribution N(0,I), effectively erasing all information about the original data x0​.5

### 1.3 The Reverse Process (Denoising): Mathematical Formulation

The generative power of diffusion models lies in learning to reverse the forward diffusion process. The goal is to learn a model pθ​ that approximates the true posterior probability distribution of the reverse transitions, q(xt−1​∣xt​).3 If the noise added at each forward step, βt​, is sufficiently small, the reverse transition q(xt−1​∣xt​) can also be shown to be Gaussian.5 However, computing q(xt−1​∣xt​) directly is intractable because it requires marginalizing over the entire data distribution.

Therefore, we parameterize the reverse process using a neural network with parameters θ. This network learns to approximate the reverse transitions with conditional Gaussian distributions:

pθ​(xt−1​∣xt​)=N(xt−1​;μθ​(xt​,t),Σθ​(xt​,t)) 3

Here, μθ​(xt​,t) and Σθ​(xt​,t) are the mean and covariance matrix of the Gaussian transition at time step t, predicted by the neural network given the current state xt​ and the time step t.

While q(xt−1​∣xt​) is intractable, the posterior distribution conditioned on the original data, q(xt−1​∣xt​,x0​), is tractable and can be derived using Bayes' theorem.5 It is also a Gaussian distribution:

q(xt−1​∣xt​,x0​)=N(xt−1​;μ~​t​(xt​,x0​),β~​t​I)

where the mean μ~​t​ and variance β~​t​ are functions of xt​, x0​, and the noise schedule parameters αt​ and βt​:

μ~​t​(xt​,x0​)=1−αˉt​αˉt−1​​βt​​x0​+1−αˉt​αt​​(1−αˉt−1​)​xt​18β~​t​=1−αˉt​1−αˉt−1​​βt​ 18

The objective during training is to make the learned reverse transitions pθ​(xt−1​∣xt​) closely match these true posterior transitions q(xt−1​∣xt​,x0​). The reverse process starts generation by sampling from a prior distribution p(xT​), which is typically set to the standard Gaussian N(0,I) (matching the distribution reached by the forward process).5 Then, it iteratively samples xt−1​ from pθ​(xt−1​∣xt​) for t=T,T−1,…,1, ultimately producing a generated sample x0​.5

### 1.4 Underlying Principles: Connections to Other Models

The theoretical underpinnings of diffusion models reveal a remarkable convergence of concepts drawn from diverse fields such as non-equilibrium thermodynamics, score matching, variational inference, and Markov decision processes.3 This confluence suggests a potentially deep and unifying mathematical framework for generative modeling.

Thermodynamics: As mentioned, the core intuition draws from non-equilibrium thermodynamics.5 The forward process mirrors the diffusion of particles from a complex, low-entropy state (data distribution) towards a high-entropy, simple equilibrium state (Gaussian noise). The learned reverse process effectively reverses entropy, guiding samples back to the structured data manifold.8

Score Matching: There is a deep connection between the denoising objective in diffusion models and score matching.10 The score of a distribution p(x) is defined as the gradient of its log-probability with respect to the data, ∇x​logp(x). It can be shown that training the neural network ϵθ​(xt​,t) to predict the noise ϵ added at step t is equivalent to learning the score function of the noisy data distribution p(xt​).14 Specifically, using Tweedie's formula, the optimal denoiser relates the noisy sample xt​ to the score: E[x0​∣xt​]∝xt​+(1−αˉt​)∇xt​​logp(xt​).18 This connection links diffusion models to score-based generative models trained using techniques like denoising score matching and simulated using methods like Langevin dynamics.8

Variational Autoencoders (VAEs): Diffusion models can be interpreted as a specific type of deep, hierarchical VAE.1 The sequence x1​,…,xT​ acts as latent variables. However, there are key distinctions: (1) The "encoder" (forward process q) is fixed and non-learned. (2) The latent variables xt​ have the same dimensionality as the original data x0​. (3) A single "decoder" network (the denoising network pθ​) is typically shared across all time steps t, conditioned on t. (4) The training objective, while derived from the Evidence Lower Bound (ELBO) common in VAEs, often uses specific reweighting or simplifications.1 This fixed forward process significantly simplifies the optimization landscape compared to standard VAEs where both encoder and decoder are learned simultaneously.1 The focus shifts entirely to learning the reverse denoising function, potentially contributing to the training stability and high sample quality observed in diffusion models.10

Energy-Based Models (EBMs): Connections also exist with EBMs. Frameworks like Diffusion by Maximum Entropy IRL (DxMI) formulate a minimax problem where an EBM provides log density estimates as reward signals to guide the diffusion model's training.20 The EBM learns an energy function whose negative value approximates the log probability density.

Reinforcement Learning (RL): The sequential generation process of diffusion models can be framed as a Markov Decision Process (MDP), where the state is (xt​,t) and the action is selecting the next state xt−1​.20 This perspective allows the application of RL techniques. For instance, Inverse Reinforcement Learning (IRL) can be used to infer a reward function from the diffusion trajectory, enabling the discovery of faster sampling paths.20 RLHF (Reinforcement Learning from Human Feedback) and related techniques like DPO (Direct Preference Optimization) are also being applied to fine-tune diffusion models based on human preferences or other reward signals, particularly for aligning model outputs (e.g., in text-to-image generation).21

The ability to frame diffusion models using concepts from these diverse fields indicates that they tap into fundamental principles of probability, optimization, and dynamics. This interrelation implies that techniques and theoretical understanding from one field might be transferable to improve diffusion models, such as using RL for faster sampling or EBMs for guidance.20

## Section 2: Architecture and Training of Diffusion Models

### 2.1 Denoising Network Architectures: The Role of U-Net Variants

The core component of the learned reverse process pθ​(xt−1​∣xt​) is the neural network that parameterizes the mean μθ​ and potentially the variance Σθ​. For tasks involving image data, the dominant architecture choice for this denoising network is the U-Net or its variants.1

The U-Net architecture, originally developed for biomedical image segmentation 16, possesses characteristics that make it particularly suitable for the denoising task in diffusion models. Its structure consists of two main paths:

Contracting Path (Encoder): This path follows a typical convolutional network structure. It comprises repeated blocks of convolutional layers (often two per block), followed by activation functions (like ReLU) and max-pooling operations.24 This path progressively downsamples the spatial resolution of the input feature maps while increasing the number of feature channels, allowing the network to capture contextual information at multiple scales.16

Expansive Path (Decoder): This path is broadly symmetric to the contracting path. It consists of up-sampling operations (e.g., transposed convolutions or bilinear upsampling followed by convolutions) that increase the spatial resolution, combined with convolutional blocks.24 Crucially, at each level of the expansive path, the feature map is concatenated with the corresponding feature map from the contracting path via skip connections.16

Skip Connections: These connections are a defining feature of the U-Net.23 They bridge the encoder and decoder paths at corresponding resolutions, allowing the decoder to directly access high-resolution features from the encoder that capture fine-grained spatial details.23 This fusion of low-level detail and high-level context is vital for the precise noise prediction required in diffusion models, mitigating the loss of spatial information common in deep networks.23

The suitability of the U-Net stems from this architecture-function synergy. The denoising network must predict noise (or the original signal) based on a noisy input xt​ and a time step t. The U-Net's encoder captures context at decreasing resolutions (analogous to processing information at increasing noise levels), while the decoder reconstructs the signal. Skip connections facilitate the flow of information across these different scales (noise levels), allowing the network to effectively combine global context with local details for accurate denoising.

Within the convolutional blocks of the U-Net used in diffusion models, common components include ResNet blocks (residual connections) to facilitate training deeper networks and self-attention layers to capture long-range dependencies within the feature maps, which can be important for modeling global structures in images.1

A critical aspect is incorporating the time step t into the network, as the denoising function must depend on the current noise level. This is typically achieved by first converting the scalar time step t into a high-dimensional embedding vector, often using sinusoidal positional embeddings (similar to those used in Transformers) or random Fourier features.1 This time embedding is then usually incorporated into the intermediate layers of the U-Net, for example, by adding it to the feature maps within the residual blocks or using it to modulate activations via adaptive normalization layers.1 This conditioning mechanism allows a single neural network ϵθ​ (or μθ​) to handle the entire reverse trajectory across all time steps t.

Research continues to explore alternative or enhanced architectures for the denoising network. Variants like U-Net++ introduce nested and dense skip connections for potentially better feature reuse.23 Other approaches replace or augment the U-Net backbone with Transformers (e.g., Diffusion Transformers or DiTs 16) or leverage concepts from continuous dynamical systems 4 to design more parameter-efficient or faster-converging networks.

### 2.2 Training Diffusion Models: Objective Functions

The goal of training a diffusion model is to adjust the parameters θ of the denoising network pθ​(xt−1​∣xt​) such that it accurately approximates the true (but intractable) reverse process transitions q(xt−1​∣xt​,x0​). This is typically framed as maximizing the likelihood of the observed data pθ​(x0​)=∫pθ​(x0:T​)dx1:T​, which is equivalent to minimizing the negative log-likelihood −logpθ​(x0​). As direct optimization of the likelihood is often intractable, training relies on optimizing a surrogate objective, specifically the Variational Lower Bound (VLB) on the log-likelihood, also known as the Evidence Lower Bound (ELBO).1 Maximizing the ELBO is equivalent to minimizing the VLB (which is the negative ELBO).

The VLB for diffusion models can be decomposed into a sum of terms, each corresponding to a step in the diffusion process 5:

LVLB​=Eq​

where:

Lt−1​=DKL​(q(xt−1​∣xt​,x0​)∥pθ​(xt−1​∣xt​)) for t=2,…,T. This term measures the Kullback-Leibler (KL) divergence between the learned reverse transition pθ​ and the true posterior q at each step. Minimizing this KL divergence forces the learned transition to match the true one.

LT​=DKL​(q(xT​∣x0​)∥pθ​(xT​)) measures the mismatch between the final noisy state distribution under the forward process and the prior assumed by the reverse process (usually N(0,I)).

L0​=−logpθ​(x0​∣x1​) represents the reconstruction likelihood of the original data given the state at t=1.

To optimize this objective, the learned mean μθ​(xt​,t) must be trained to predict the true posterior mean μ~​t​(xt​,x0​). A common and effective parameterization, proposed by Ho et al. (DDPM) 10, involves training the neural network, denoted ϵθ​(xt​,t), to predict the noise component ϵ that was added to x0​ to obtain xt​ via the relation xt​=αˉt​​x0​+1−αˉt​​ϵ.3 The learned mean μθ​ can then be expressed in terms of the predicted noise ϵθ​:

μθ​(xt​,t)=αt​​1​(xt​−1−αˉt​​βt​​ϵθ​(xt​,t)) 3 (Note: βt​=1−αt​)

The variance Σθ​(xt​,t) is often kept fixed rather than learned, typically set to σt2​I where σt2​ is either βt​ or the true posterior variance β~​t​.5 Fixing the variance simplifies the model and training. However, learning the variance is also possible and can sometimes improve likelihoods.1

Substituting the noise prediction parameterization into the KL divergence term Lt−1​ leads to a loss term involving the squared error between the true noise ϵ and the predicted noise ϵθ​, weighted by terms depending on the noise schedule.5 However, Ho et al. 10 empirically found that a simplified objective function, which essentially ignores the complex weighting factors from the VLB and directly minimizes the Mean Squared Error (MSE) between the true and predicted noise, performs very well, often yielding better sample quality.3 This simplified loss is widely used:

$$L_\text{simple} = \mathbb{E}_{t \sim \mathcal{U}(1, T), \mathbf{x}_0 \sim q(\mathbf{x}_0), \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \Big$$ 3

The training algorithm typically proceeds as follows (e.g., Algorithm 1 in DDPM 3):

Repeat until convergence:

Sample a batch of data points x0​ from the training dataset.

For each data point, sample a time step t uniformly from {1,…,T}.

Sample a noise vector ϵ from N(0,I).

Compute the noisy sample xt​=αˉt​​x0​+1−αˉt​​ϵ.

Feed xt​ and t into the neural network to predict the noise ϵθ​(xt​,t).

Calculate the loss Lsimple​ (or a variant) based on ∥ϵ−ϵθ​∥2.

Compute gradients and update the network parameters θ using an optimizer (e.g., Adam). 3

This simplified objective highlights a pragmatic aspect often seen in deep learning: while the VLB provides theoretical justification, the core task the network needs to master – predicting the noise (or equivalently, the denoised signal) – can be effectively optimized using a simpler, more direct objective.11 Empirical performance and training stability often guide the choice of the final loss function.11 It is possible to add auxiliary loss terms (e.g., for regularization or enforcing specific properties) 28, but care must be taken as this might complicate training or deviate significantly from the underlying probabilistic model.28 Training can sometimes exhibit oscillatory loss behavior, particularly as the network quickly learns the easier task of denoising at high noise levels (t≫1) while struggling with low noise levels (t≈1).31

### 2.3 Sampling Procedure: Iterative Denoising for Generation

Once the denoising network pθ​ (parameterized as ϵθ​) is trained, it can be used to generate new data samples. The sampling process starts from pure noise and iteratively applies the learned reverse transitions to gradually denoise it into a sample from the target data distribution.3

The generation algorithm (e.g., Algorithm 2 in DDPM 3) proceeds as follows:

Initialization: Start by sampling an initial noise vector xT​ from the prior distribution, typically xT​∼N(0,I).

Iterative Denoising Loop: Iterate backwards through time steps from t=T down to t=1: a. Sample a random Gaussian noise vector z∼N(0,I) (if t>1; set z=0 if t=1). b. Use the trained neural network to predict the noise component ϵθ​(xt​,t) based on the current state xt​ and time step t. c. Calculate the mean of the reverse transition pθ​(xt−1​∣xt​) using the predicted noise: μθ​(xt​,t)=αt​​1​(xt​−1−αˉt​​βt​​ϵθ​(xt​,t)) 3 d. Determine the variance σt2​ for the reverse step. Common choices are σt2​=βt​ or σt2​=β~​t​=1−αˉt​1−αˉt−1​​βt​.5 e. Sample the state at the previous time step xt−1​ from the Gaussian distribution: xt−1​=μθ​(xt​,t)+σt​z 3

Output: After iterating through all time steps, the final state x0​ is the generated sample from the model. 5

A crucial characteristic of this sampling process is its iterative nature. It typically requires evaluating the neural network once for each time step t from T down to 1. Since T is often large (e.g., 1000 or even more 1), this sequential evaluation leads to slow sampling speeds compared to models that generate samples in a single forward pass, like standard GANs or VAEs.1 This high computational cost during inference is a primary drawback of diffusion models and represents a major bottleneck for their practical application, especially in latency-sensitive scenarios. Consequently, significant research effort has been dedicated to developing techniques for accelerating the sampling process. These include designing more efficient deterministic samplers based on Ordinary Differential Equation (ODE) formulations (like DDIM - Denoising Diffusion Implicit Models) 19, using advanced SDE solvers 19, learning optimized or shorter sampling schedules 1, knowledge distillation to train faster "student" models 4, and exploring non-Markovian sampling approaches.38 The goal is to reduce the number of required network evaluations (sampling steps) significantly without substantially degrading the quality of the generated samples.

## Section 3: Adapting Diffusion Models for Discrete Data (Text)

### 3.1 The Challenge: Continuous vs. Discrete Domains

The foundational diffusion models, such as Denoising Diffusion Probabilistic Models (DDPMs) and Noise Conditioned Score Networks (NCSNs), were primarily developed for continuous data domains like images and audio.2 These models rely heavily on the properties of continuous state spaces and the use of Gaussian noise for both the forward corruption process and the parameterization of the reverse denoising process.

However, natural language text presents a fundamental challenge: it is inherently discrete.9 Text is composed of sequences of discrete units (tokens), such as words, subwords, or characters, drawn from a finite vocabulary. Applying Gaussian noise directly to token indices or one-hot vector representations is mathematically ill-defined and lacks semantic meaning. This discrete nature creates a mismatch with the standard continuous diffusion framework, necessitating specific adaptations to apply diffusion principles to text generation.32 Research has primarily explored two main strategies to bridge this gap: performing diffusion in a continuous embedding space or defining diffusion processes directly on the discrete token space.

### 3.2 Approach 1: Diffusion in Continuous Embedding Space

This approach seeks to leverage the well-established machinery of continuous Gaussian diffusion models by transforming the discrete text problem into a continuous one.9 The core idea involves three steps:

Embedding: Map the sequence of discrete tokens w=(w1​,…,wn​) into a sequence of continuous vectors z0​=(z0,1​,…,z0,n​) residing in an embedding space Rd. This is done using an embedding function EMB(wi​) for each token wi​.9 The embeddings can be pre-trained (e.g., Word2Vec, GloVe) or, more effectively, learned jointly with the diffusion model parameters during end-to-end training, as demonstrated in Diffusion-LM.41 The initial state z0​ might be defined via a transition q(z0​∣w)=N(z0​;EMB(w),σ02​I).41

Continuous Diffusion: Apply the standard Gaussian forward diffusion process to the sequence of embeddings z0​ over T steps, generating noisy latent variables z1​,…,zT​. Subsequently, train a denoising network (often Transformer-based for sequences 41) pθ​(zt−1​∣zt​) to reverse this process in the continuous embedding space.9

Rounding/Mapping Back: After the reverse process generates a final continuous embedding sequence z0​, map these vectors back to a sequence of discrete tokens w.33 This step is critical and poses a significant challenge. A simple approach is to find the nearest word embedding in the vocabulary for each vector in z0​. However, the generated z0​ might not perfectly align with any existing word embedding, leading to "rounding errors" that can degrade the quality and fluency of the generated text.33 More sophisticated techniques include training a dedicated rounding module (e.g., a linear layer followed by softmax over the vocabulary) as part of the end-to-end objective 41 or employing tricks during decoding, such as the "clamping trick" proposed in Diffusion-LM, which forces intermediate predictions towards the nearest valid word embeddings during the denoising steps.41

Prominent examples of models using this approach include Diffusion-LM 9, DiffuSeq (which applies partial noising for conditional generation) 9, GENIE 47, Plaid 45, and Difformer.42

While this approach allows leveraging powerful continuous diffusion techniques, it faces challenges. Jointly learning the embedding space can be unstable and may lead to embedding collapse; techniques like the anchor loss have been proposed to mitigate this.42 The quality of the learned embeddings is crucial 52, and the rounding step remains a potential source of errors, impacting the final text quality.33

### 3.3 Approach 2: Discrete Diffusion Processes

An alternative strategy is to define the diffusion process directly within the discrete state space of tokens, avoiding the need for a continuous embedding space and the associated rounding issues.21

Discrete Noise / Corruption: Instead of adding continuous Gaussian noise, these models employ discrete corruption mechanisms. The forward process gradually corrupts the original token sequence x0​. Common corruption types include:

Uniform Noise (Multinomial Diffusion): At each step, tokens are randomly replaced with other tokens sampled uniformly from the vocabulary.38

Masking (Absorbing Diffusion): Tokens are replaced with a special token with a certain probability.37 The token often acts as an absorbing state, meaning once a token is masked, it stays masked.37 Masking-based diffusion has often shown superior performance compared to other discrete noise types.58

Transition Matrices: The forward process can be formalized using time-dependent Markov transition matrices Qt​∈RK×K, where K is the vocabulary size.41 If xt−1​ is represented as a one-hot vector, the distribution of the next state xt​ is given by q(xt​∣xt−1​)=Categorical(xt​;Qt​xt−1​).58 The matrix Qt​ encodes the probabilities of transitioning from any token j to any token i (or the `` state) at step t.

Reverse Process: The learned reverse process pθ​(xt−1​∣xt​) aims to predict the probability distribution over the vocabulary for each position at the previous, less corrupted step t−1, given the current state xt​ and time t.54 This often involves predicting the original token x0​ and deriving the reverse transition probabilities from it.

Loss Function: Training typically involves minimizing a loss function that measures the discrepancy between the predicted distribution and the true target state (either xt−1​ or x0​). Common choices include the cross-entropy loss 53 or objectives derived from the variational lower bound (ELBO) adapted for discrete spaces.57 Specialized loss functions like score entropy have also been proposed.53

The D3PM (Discrete Denoising Diffusion Probabilistic Models) framework provides a general formulation for these models, encompassing various noise types and transition structures.38 Other examples include DiffusionBERT 50, SEDD 45, Diffusion-NAT 37, and UniDisc.53

The main advantage of discrete diffusion is that it operates directly on the natural representation of text, avoiding the problematic continuous-to-discrete conversion and potential rounding errors.37 It can be more naturally integrated with token-level operations and pre-trained models that operate on discrete tokens.37 However, this area is generally less mature than continuous diffusion. Designing effective discrete noise processes, transition matrices, and loss functions requires careful consideration, and these models might not directly benefit from the geometric intuitions available in continuous embedding spaces.42

This fundamental dichotomy between continuous embedding space diffusion and discrete state-space diffusion represents a core research tension. Continuous approaches leverage well-understood Gaussian processes but face the interface problem, while discrete approaches are more natural for text but require developing new diffusion formalisms and may lose the richness of embedding spaces.

### 3.4 Hybrid and Advanced Techniques

Beyond the two primary approaches, several hybrid and advanced techniques aim to combine their benefits or overcome their limitations, often driven by the need for improved quality and, critically, faster sampling speeds.

Flow Matching: Inspired by continuous flow matching, techniques like Discrete Flow Matching 40 and Conditional Flow Matching aim to learn a simpler, often straighter, path between the noise distribution and the data distribution. This allows for generation in fewer steps (potentially even a single step) compared to standard diffusion.40 FlowSeq is an example applied to text generation.40 This focus on efficiency reflects the practical limitations imposed by the multi-step generation process of standard diffusion.32

Soft Absorbing States: This technique attempts to bridge the gap between continuous and discrete diffusion by incorporating learnable "soft mask" embeddings.44 Instead of replacing tokens with a fixed `` token, it replaces their continuous representations with a learnable vector. This allows the model to remain in a continuous space while incorporating the idea of masking from discrete diffusion, potentially gaining benefits from both paradigms, including faster convergence and sampling.44

Integrating Pre-trained Language Models (PLMs): A significant trend involves leveraging the power of large PLMs (like BERT, T5, BART, GPT variants) within the diffusion framework.37 This is particularly relevant for discrete diffusion. For example, Diffusion-NAT uses BART's encoder-decoder architecture as the denoising network, aligning the discrete diffusion task (recovering masked tokens) with BART's masked language modeling pre-training objective.37 Other works explore adapting pre-trained AR models for diffusion tasks through continual pre-training.48 This strategy injects strong linguistic priors into the diffusion model, potentially reducing the need for massive diffusion-specific training data and improving text coherence.37

Custom Noise Processes: Researchers are exploring noise processes beyond simple uniform replacement or masking. DiffusER, for instance, uses edit-based operations (insertions, deletions) as the corruption process.50 Using learned soft masks instead of hard masks is another variation.44

Continuous Time Formulation: Extending discrete diffusion models to operate in continuous time, often using the formalism of Continuous-Time Markov Chains (CTMCs), offers greater flexibility in the sampling process.39 This allows for choosing an arbitrary number of steps during inference without retraining, enabling a trade-off between computational cost and sample quality.55

These advanced techniques highlight the ongoing effort to optimize diffusion models for text, focusing on improving generation quality, enhancing controllability, and critically, overcoming the inherent slowness of the iterative sampling process.

## Section 4: Diffusion Language Models (Diffusion LLMs)

### 4.1 Defining Diffusion LLMs: Models Applying Diffusion to Text Generation

The term "Diffusion Language Model" (Diffusion LM) or "Diffusion LLM" does not refer to a single, specific architecture in the way "Transformer LLM" typically does. Instead, it serves as an umbrella term for any language model that employs a diffusion-based mechanism – involving iterative noising and subsequent learned denoising or reconstruction – as its fundamental generative principle.9

The defining characteristic of these models is their iterative refinement process. Unlike standard autoregressive (AR) LLMs, which generate text sequentially token by token from left to right 9, Diffusion LLMs typically start with a representation of the entire sequence (either random noise or a fully masked sequence) and progressively refine this representation over multiple steps until a coherent text sequence emerges.9 This generation process is usually non-autoregressive (NAR) in nature, meaning that updates within each refinement step can often be computed in parallel across the entire sequence length.9 This core operational difference marks a significant paradigm shift from traditional AR language modeling.

### 4.2 Key Architectures and Models

The landscape of Diffusion LLMs is diverse, reflecting the various strategies developed to adapt diffusion principles for discrete text data, as discussed in Section 3. This variety indicates an active and experimental phase of research rather than a single converged architecture. Key examples can be categorized based on their underlying approach:

Continuous Embedding Space Models: These models perform Gaussian diffusion on sequences of word embeddings.

Diffusion-LM: A pioneering work that introduced end-to-end training of embeddings and the diffusion model, along with the clamping trick for rounding.9

DiffuSeq: Focuses on sequence-to-sequence tasks by applying partial noising only to the target sequence embeddings, using the source embedding as a condition.9

Plaid: A large-scale (1.3B parameters) continuous diffusion LM trained from scratch, demonstrating scaling potential.45

GENIE: Another large-scale pre-trained diffusion LM for sequence-to-sequence tasks.47

Difformer: Employs a Transformer architecture for denoising embeddings and introduces techniques like anchor loss and noise rescaling to address optimization challenges.42

Discrete State Space Models: These models define diffusion directly on token indices or one-hot vectors.

D3PM Framework: Provides a general mathematical framework for discrete diffusion using various transition matrices, including multinomial noise and absorbing (masking) states.38

Masked Diffusion Models: A successful specialization within D3PM using masking as the corruption process. Often uses Transformer backbones for denoising.53

DiffusionBERT: Improves generative masked language models by incorporating diffusion principles.50

SEDD (Score Entropy Discrete Diffusion): Introduces a novel loss function for discrete diffusion.45

UniDisc: A unified multimodal discrete diffusion model capable of jointly processing and generating text and images using a shared vocabulary and masking.53

PLM-Integrated Models: These leverage pre-trained language models within the diffusion framework.

Diffusion-NAT: Uses a pre-trained BART model as the denoiser in a discrete (masking-based) diffusion process for non-autoregressive text generation.37

Adapted AR Models: Approaches that adapt existing pre-trained AR models (like GPT variants) into diffusion models via techniques like continual pre-training.48

Flow-Based Models: Utilize flow matching principles for potentially faster, fewer-step generation.

FlowSeq: Applies flow matching to sequence-to-sequence text generation.40

Discrete Flow Matching Models: Generalize flow matching to discrete state spaces, showing promising results.50

Other Conceptual Models:

Latent Diffusion for Text: Applies diffusion to learned latent semantic embeddings representing paragraphs, combined with an AR decoder (e.g., RDM - Reparameterized Diffusion Model).33

Diffusion-of-Thought (DoT): Integrates diffusion with Chain-of-Thought prompting for improved reasoning, allowing reasoning steps to diffuse iteratively.45

This wide array of approaches underscores that "Diffusion LLM" currently represents a collection of related but distinct techniques exploring the application of iterative refinement to language, rather than a monolithic model class.

### 4.3 Training Diffusion LLMs: Specific Objectives and Considerations

The training process for Diffusion LLMs varies depending on whether they operate in continuous or discrete space and whether they integrate PLMs.

Continuous Models: Training typically involves optimizing an objective derived from the VLB, adapted for the embedding space.

End-to-End Objective (e.g., Diffusion-LM): Jointly learns the word embeddings EMB(w) and the diffusion model parameters θ. The objective includes terms for the diffusion process (often a simplified MSE loss on predicted noise or embeddings) and a term for the rounding step (e.g., minimizing reconstruction error or maximizing likelihood of original words given final embedding).41 Stability can be a concern due to the learnable data distribution (embeddings); techniques like anchor loss aim to regularize the embedding space.42

Conditional Models (e.g., DiffuSeq): Use partial noising, applying noise only to target sequence embeddings. The training objective minimizes the reconstruction error for the target part, implicitly conditioned on the source part via the shared network (e.g., Transformer attention).9

Discrete Models: Training focuses on learning the reverse transitions in the discrete token space.

Predicting Original Tokens: A common objective is to train the model pθ​(xt−1​∣xt​) to predict the original clean token sequence x0​ given the noisy input xt​. The loss is often a cross-entropy loss summed over all positions and time steps, comparing the predicted distribution over the vocabulary with the true token x0​.53

VLB-based Objectives: It's also possible to derive and optimize the ELBO/VLB directly for discrete processes, leading to objectives involving KL divergences between categorical distributions or auxiliary cross-entropy terms.57

Specialized Losses: Methods like SEDD introduce novel loss functions like score entropy tailored for discrete diffusion.53

PLM Integration: When integrating PLMs, the training objective is often adapted to align with the PLM's pre-training task.

Masked Token Recovery (e.g., Diffusion-NAT): The diffusion model (using BART as backbone) is trained to predict the original tokens for all masked positions in the noised input, minimizing a negative log-likelihood objective similar to BART's masked language modeling objective.37

Continual Pre-training: Existing AR models can be adapted by continually pre-training them on a diffusion-specific objective, potentially bridging the gap between AR prediction and denoising tasks.48

General Considerations: Regardless of the specific approach, training Diffusion LLMs involves careful consideration of the noise schedule (βt​ or discrete transition matrices Qt​) 67, which significantly impacts performance. Like other large generative models, they can be data-hungry 68, although integrating PLMs might alleviate this by leveraging existing knowledge.37 Hyperparameter tuning (learning rates, batch sizes, schedule parameters) can also be sensitive.68

### 4.4 Text Generation Process in Diffusion LLMs: Step-by-Step Denoising/Reconstruction

The inference or text generation process in Diffusion LLMs follows the general principle of reversing the diffusion process, starting from a random or fully corrupted state and iteratively refining it using the learned denoising network.9 The specifics vary based on the model type:

Initialization:

Continuous Models: Start with a sequence of random vectors zT​ sampled from a standard Gaussian distribution N(0,I).41 The sequence length needs to be determined beforehand.

Discrete Models (Masking-based): Start with a sequence xT​ consisting entirely of `` tokens.37

Iterative Refinement Loop: Iterate from t=T down to t=1 (or fewer steps if using acceleration techniques):

Input: Provide the current state (zt​ or xt​) and the time step t to the denoising network pθ​. If the model is conditional, also provide the conditioning information (e.g., source sentence embedding, prompt).9

Prediction: The network predicts the parameters of the previous state distribution pθ​(xt−1​∣xt​).

Continuous: Predicts the noise ϵθ​(zt​,t) or the mean μθ​(zt​,t).41

Discrete: Predicts the probability distribution over the vocabulary for each position, often by first predicting the clean state x0​ and deriving the reverse transition probabilities.37

Sampling: Sample the next state xt−1​ (or zt−1​) from the predicted distribution.

Continuous: Sample zt−1​∼N(zt−1​;μθ​(zt​,t),σt2​I).41 May involve clamping intermediate predictions to nearest embeddings.41

Discrete: Sample tokens for each position based on the predicted categorical distributions.37

Final Output:

Continuous Models: The final state z0​ is mapped back to discrete tokens using the chosen rounding method.41

Discrete Models: The final state x0​ is the generated token sequence.37

Conditional Generation: Conditioning is typically handled by feeding the conditional input (e.g., source text embedding, class label embedding) to the denoising network at every step t, allowing it to influence the prediction.9 Classifier-free guidance is a common technique where the model is trained both conditionally and unconditionally, and the generation is guided by extrapolating between the conditional and unconditional predictions.2

Parallelism: A key aspect is that within each step t, the predictions and updates for all positions in the sequence can often be computed in parallel using architectures like Transformers.9 This contrasts with the inherently sequential nature of AR generation. This non-autoregressive, iterative refinement process fundamentally distinguishes Diffusion LLMs from AR Transformers, opening up possibilities for different generation dynamics, global planning, and control mechanisms.45

## Section 5: Comparative Analysis: Diffusion LLMs vs. Transformer LLMs

Comparing Diffusion Language Models with the dominant Transformer-based Autoregressive (AR) Language Models reveals a complex landscape of trade-offs involving generation paradigms, computational efficiency, output quality, and control capabilities. Understanding these differences is crucial for assessing the potential roles of diffusion models in the future of language generation.

### 5.1 Conceptual Differences: Generation Paradigm

The most fundamental difference lies in the generation process itself:

Diffusion LLMs: Employ an iterative refinement strategy. They typically start from a non-informative state (Gaussian noise for continuous models, masked sequences for discrete models) representing the entire sequence and progressively denoise or reconstruct it over multiple steps. This process is generally non-autoregressive (NAR), allowing parallel updates across the sequence within each step.9 The core task learned is denoising or reconstruction.

Transformer (AR) LLMs: Generate text sequentially, one token at a time, conditioning each prediction on the previously generated tokens (typically in a strict left-to-right order).9 The core task learned is next-token prediction.

### 5.2 Potential Strengths of Diffusion LLMs

The iterative, non-autoregressive nature of diffusion models offers several potential advantages over AR models:

Controllability: The multi-step refinement process provides multiple opportunities to inject control signals or guidance. This makes diffusion models potentially better suited for tasks requiring fine-grained control over semantic content, style, structure (e.g., syntax), or other attributes.32 Techniques like classifier-free guidance 50 can be readily applied. The ability to perform joint inpainting or editing across modalities is also a potential benefit.53

Diversity: Diffusion models may naturally produce more diverse outputs compared to AR models, which can sometimes fall into repetitive generation patterns.5 They offer mechanisms to control the trade-off between sample quality and diversity during sampling.53

Non-Autoregressive Properties:

Parallelism: Computations within each denoising step can be parallelized across the sequence length, offering potential throughput advantages.9

Flexibility: The NAR nature allows for generation orders other than left-to-right, such as filling in masked portions of text (e.g., FiLM 50) or generating tokens in arbitrary orders.48

Potential Speed: While multi-step latency is high, the parallel computation per step means that for very long sequences, NAR generation could theoretically be faster than sequential AR generation if the number of required diffusion steps (Teff​) is significantly smaller than the sequence length (L).32

Self-Correction/Refinement: The iterative process inherently allows the model to revisit and potentially correct errors or inconsistencies introduced in earlier steps.33 This contrasts with AR models where errors can accumulate irreversibly. This may help mitigate the "exposure bias" problem seen in AR models, where the model is trained on ground-truth prefixes but tested by generating its own, potentially erroneous, prefixes.33

Theoretical Foundations: Connections to score matching, thermodynamics, and VAEs provide alternative theoretical frameworks for understanding and improving generative modeling.1

### 5.3 Potential Weaknesses/Challenges of Diffusion LLMs

Despite their potential, diffusion models currently face significant challenges, particularly in the language domain:

Computational Cost / Sampling Speed: This is arguably the most significant practical drawback. The need for multiple sequential evaluations of the denoising network (often T=1000 or more in original formulations, though acceleration techniques aim for Teff​ in the tens or low hundreds) results in high sampling latency compared to the single-pass generation of AR models.1 Even with acceleration, sampling is often considerably slower than optimized AR inference. Training can also be computationally intensive.68 This efficiency gap is a major hurdle for widespread adoption.

Generation Quality (Fluency/Coherence): While improving, current Diffusion LLMs often lag behind state-of-the-art AR models (especially large-scale ones) in generating highly fluent, coherent, and contextually appropriate text, particularly for longer sequences or complex tasks.33 Performance is sensitive to the number of sampling steps, with quality degrading significantly as steps are reduced to improve speed.39

Handling Discrete Data: As discussed in Section 3, adapting continuous diffusion processes to discrete text remains a core challenge. Continuous embedding models suffer from rounding errors and embedding space stability issues 33, while discrete diffusion models require careful design of corruption processes and loss functions and are less mature.35

Scalability: Transformer-based AR models have proven remarkably scalable, with performance consistently improving with model size and data. While initial studies suggest diffusion models also exhibit scaling properties 50, it remains less established whether they can reach the performance of multi-trillion parameter AR models on language tasks.45 AR models currently dominate the very large language model landscape.

Likelihood Evaluation: Calculating the exact likelihood pθ​(x0​) is often intractable for diffusion models, making direct comparison with AR models using standard metrics like perplexity difficult or require approximations.19 While methods exist to estimate or bound the likelihood, they add complexity.

### 5.4 Training Efficiency and Data Requirements

Diffusion LLMs: Training involves sampling noise levels and noise vectors for each data point in a batch and optimizing the denoising objective (e.g., MSE or cross-entropy). While potentially more stable than GAN training 5, it might be slower per epoch than AR training due to the sampling overhead. Diffusion models can be data-hungry, requiring large datasets to learn complex distributions.68 However, integrating PLMs can potentially reduce this requirement by leveraging pre-existing linguistic knowledge.37

Transformer (AR) LLMs: Training uses the standard and highly efficient next-token prediction objective (maximizing likelihood via cross-entropy). The pre-training paradigm relies on massive unlabeled text corpora, which has proven extremely effective for learning general language understanding and generation capabilities.

### 5.5 Inference Speed and Latency

Diffusion LLMs: Inference latency is primarily determined by the number of sequential denoising steps (Teff​) multiplied by the time taken for one forward pass of the denoising network. Parallel computation within each step improves throughput for batch processing but does not reduce the latency for generating a single sample.32 Acceleration techniques that reduce Teff​ are essential for practical use.1 For very long sequences, if Teff​ can be made small enough, NAR diffusion could potentially be faster than sequential AR generation.62

Transformer (AR) LLMs: Inference latency scales linearly with the length of the sequence to be generated, as tokens are produced one after another. Techniques like KV-caching significantly optimize this process by reusing intermediate computations.62 AR models are generally very fast for generating short-to-moderate length sequences but become slower for extremely long outputs.

### 5.6 Generation Quality and Controllability

Diffusion LLMs: Offer high potential for controllability due to the iterative process allowing for guidance injection 32 and potentially greater output diversity.63 However, current models often exhibit lower quality (fluency, coherence, factual accuracy) compared to the best AR models, especially on complex prompts or long generations.33 Quality is strongly dependent on the number of sampling steps used.39

Transformer (AR) LLMs: Represent the state-of-the-art in terms of raw text quality, fluency, coherence, and in-context learning capabilities, particularly at large scales.39 Control is typically achieved indirectly through prompting techniques or fine-tuning.70 Known weaknesses include susceptibility to generating repetitive text, factual inaccuracies (hallucinations), and sensitivity to prompt phrasing.33

The comparison highlights a clear trade-off landscape. Diffusion LLMs present compelling theoretical advantages, particularly regarding control, diversity, and the potential to overcome limitations inherent in the AR paradigm. However, they currently face significant practical challenges in sampling efficiency and achieving parity with the generation quality of highly refined, scaled-up AR Transformers. The choice between these paradigms depends heavily on the specific application's priorities – whether fine-grained control and diversity are paramount, or whether maximum fluency and speed are the primary requirements.

### 5.7 Comparative Summary Table

The following table summarizes the key differences and trade-offs between Diffusion LLMs and Transformer-based AR LLMs:

Feature | Diffusion LLMs | Transformer (AR) LLMs
--- | --- | ---
Generation Paradigm | Iterative Refinement, Non-Autoregressive (NAR) | Sequential Prediction, Autoregressive (AR)
Core Task Learned | Denoising / Reconstruction | Next-Token Prediction
Training Stability | Generally Stable | Stable
Sampling Speed/Latency | Slow (Multi-step, Teff​ dependent) | Fast (Seq. Length Dependent, KV-caching)
Parallelism (Inference) | Parallel within each step | Sequential token generation
Controllability | Potentially High (Iterative Guidance) | Indirect (Prompting, Fine-tuning)
Generation Quality | Improving, often lower fluency/coherence currently | State-of-the-Art Fluency/Coherence
Diversity | Potentially High | Moderate (Risk of Repetition)
Handling Discrete Data | Inherent Challenge (Embedding/Rounding or Discrete Noise) | Native (Operates directly on tokens)
Scalability | Under Investigation, Promising | Proven at Massive Scale
Likelihood Calculation | Often Intractable or Requires Approximation | Straightforward (Perplexity)
Key Strengths | Control, Diversity, NAR Flexibility, Error Correction Potential | Quality, Scalability, Training/Inference Efficiency
Key Weaknesses | Sampling Speed, Current Quality Gap, Discrete Data Handling | Direct Control, Repetition, Exposure Bias, Hallucination

## Section 6: Conclusion and Future Directions

### 6.1 Summary of Diffusion Models and their Role in Language Generation

Diffusion models represent a distinct and increasingly influential paradigm in generative modeling. Rooted in the concept of reversing a gradual noising process, they operate via iterative refinement, transforming simple noise into complex data samples. Their application to natural language processing, however, necessitates overcoming the fundamental mismatch between the continuous nature of standard diffusion processes and the discrete nature of text. This has led to two primary adaptation strategies: performing diffusion in continuous embedding spaces, which leverages Gaussian diffusion but introduces rounding challenges, and developing novel discrete diffusion processes that operate directly on tokens using mechanisms like masking.

Models employing these techniques, collectively termed Diffusion Language Models (Diffusion LLMs), offer a non-autoregressive alternative to traditional Transformer-based AR LLMs. They generate text by iteratively refining an initial representation (noise or masks) over multiple steps. This iterative, often parallel, refinement process contrasts sharply with the sequential, token-by-token generation of AR models. The comparison between these paradigms reveals significant trade-offs: Diffusion LLMs hold theoretical promise for enhanced controllability, output diversity, and potentially overcoming AR limitations like exposure bias, but currently face practical hurdles in terms of sampling speed and achieving the state-of-the-art fluency and coherence demonstrated by large-scale AR models. The trend of integrating PLMs into diffusion frameworks suggests a potential path towards combining the strengths of both paradigms.

### 6.2 Current Research Frontiers and Open Challenges

The field of Diffusion LLMs is rapidly evolving, with numerous active research frontiers aimed at overcoming current limitations and unlocking their full potential. Key challenges and future directions include:

Sampling Efficiency: This remains arguably the most critical bottleneck.1 Research continues to focus on developing faster ODE/SDE solvers, optimized sampling schedules, knowledge distillation techniques, flow-matching approaches, and non-Markovian methods to drastically reduce the number of required network evaluations (Teff​) without sacrificing generation quality.1 The future viability of diffusion models as mainstream LLMs hinges critically on solving this efficiency problem.

Scaling and Quality Improvement: Closing the performance gap with leading AR models requires scaling Diffusion LLMs to larger parameter counts and training datasets.45 Understanding the scaling laws specific to diffusion architectures for language is crucial.62 Continued work is needed to improve the fluency, coherence, factual accuracy, and long-range dependency handling of generated text.33

Refining Discrete Data Handling: Further innovation is needed in both primary adaptation strategies. For continuous embedding models, research focuses on developing more stable training methods (e.g., anchor loss 42), learning better embedding spaces 52, and reducing rounding errors.41 For discrete diffusion models, efforts concentrate on designing more effective transition kernels (noise types), theoretically grounded loss functions, and architectures that better capture linguistic structure.53

Reasoning, Planning, and Control: While quality and speed are current challenges, the unique iterative refinement mechanism holds significant, largely untapped potential. Exploring the capabilities of diffusion models for complex reasoning (e.g., Diffusion-of-Thought 45), long-range planning, and fine-grained controllable generation represents a promising avenue.22 This is particularly relevant for tasks where purely AR approaches struggle due to their sequential nature.33 Diffusion might carve out strong niches in these areas. Techniques for alignment, such as RLHF and DPO, are also being actively adapted for diffusion models.22

Likelihood Estimation and Evaluation: Developing robust and efficient methods for calculating or accurately estimating likelihoods for diffusion models remains important for rigorous evaluation and comparison with AR models based on metrics like perplexity.19

Multimodality: Extending diffusion frameworks to seamlessly handle multiple modalities, including text, images, audio, and video, within a unified model (e.g., UniDisc 53) is a major direction for building more versatile AI systems.35

Theoretical Understanding: Further theoretical work is needed to deepen our understanding of the properties of diffusion processes, especially for discrete data, and their connections to other generative modeling paradigms.50

In conclusion, diffusion models offer a compelling alternative generative paradigm for language, characterized by iterative refinement and non-autoregressive potential. While facing significant challenges, particularly concerning sampling efficiency and current quality benchmarks compared to scaled AR Transformers, ongoing research into architectural innovations, training techniques, acceleration methods, and hybrid approaches holds promise for addressing these limitations. The potential for enhanced control and applicability to complex reasoning tasks suggests that diffusion models are likely to play an increasingly important role in the evolving landscape of language generation, potentially leading to a convergence of ideas where future models blend the strengths of iterative refinement and powerful pre-trained representations.



Works cited

http://www.cs.unc.edu , accessed on April 18, 2025, https://www.cs.unc.edu/~ronisen/teaching/fall_2022/pdf_lectures/lecture7-8_diffusion_model.pdf

Diffusion Models and Representation Learning: A Survey - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2407.00783v1 

How diffusion models work: the math from scratch | AI Summer, accessed on April 18, 2025, https://theaisummer.com/diffusion-models/ 

The Missing U for Efficient Diffusion Models - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2310.20092v4 

What are Diffusion Models? | Lil'Log, accessed on April 18, 2025, https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ 

Lecture Notes in Probabilistic Diffusion Models - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2312.10393v1 

Step by Step visual introduction to Diffusion Models. - Blog by Kemal Erdem, accessed on April 18, 2025, https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models/ 

Diffusion model - Wikipedia, accessed on April 18, 2025, https://en.wikipedia.org/wiki/Diffusion_model 

http://arxiv.org , accessed on April 18, 2025, https://arxiv.org/pdf/2210.08933

proceedings.neurips.cc, accessed on April 18, 2025, https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

Variational Autoencoders and Diffusion Models - CS231n, accessed on April 18, 2025, https://cs231n.stanford.edu/slides/2023/lecture_15.pdf

What is the Reverse Diffusion Process? - Analytics Vidhya, accessed on April 18, 2025, https://www.analyticsvidhya.com/blog/2024/07/reverse-diffusion-process/ 

Introduction to Diffusion Models for Machine Learning | SuperAnnotate, accessed on April 18, 2025, https://www.superannotate.com/blog/diffusion-models 

Introduction to Diffusion Models for Machine Learning - AssemblyAI, accessed on April 18, 2025, https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction 

An In-Depth Guide to Denoising Diffusion Probabilistic Models DDPM – Theory to Implementation - LearnOpenCV, accessed on April 18, 2025, https://learnopencv.com/denoising-diffusion-probabilistic-models/ 

Diffusion Transformer (DiT) Models: A Beginner's Guide - Encord, accessed on April 18, 2025, https://encord.com/blog/diffusion-models-with-transformers/ 

An Introduction to Diffusion Models and Stable Diffusion - Marvik - Blog, accessed on April 18, 2025, https://blog.marvik.ai/2023/11/28/an-introduction-to-diffusion-models-and-stable-diffusion/ 

Lecture 15. Diffusion Models - http://Math.Utah.Edu , accessed on April 18, 2025, https://www.math.utah.edu/~bwang/mathds/Lecture15.pdf

Diffusion Models: A Comprehensive Survey of Methods and Applications, accessed on April 18, 2025, https://accesson.kisti.re.kr/upload2/article/originPdf/004538/ATN0045381127.pdf

Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models - NeurIPS, accessed on April 18, 2025, https://proceedings.neurips.cc/paper_files/paper/2024/file/2bed6c14cd5ea97a9bc1e6094941bde7-Paper-Conference.pdf

REVIEW - Oxford Academic, accessed on April 18, 2025, https://academic.oup.com/nsr/article-pdf/11/12/nwae348/61414473/nwae348.pdf

xie-lab-ml/awesome-alignment-of-diffusion-models - GitHub, accessed on April 18, 2025, https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models 

Unet Architecture for Diffusion Models | Restackio, accessed on April 18, 2025, https://www.restack.io/p/ai-diffusion-educational-resources-answer-unet-architecture 

Unet Architecture Explained | Restackio, accessed on April 18, 2025, https://www.restack.io/p/ai-diffusion-educational-resources-answer-unet-architecture-explained 

U-Net - Wikipedia, accessed on April 18, 2025, https://en.wikipedia.org/wiki/U-Net 

The U-Net (actually) explained in 10 minutes - YouTube, accessed on April 18, 2025, https://m.youtube.com/watch?v=NhdzGfB1q74&pp=ygUSI21hbWFiYW5hdW5ldGFyaWth

Train a diffusion model - Hugging Face, accessed on April 18, 2025, https://huggingface.co/docs/diffusers/tutorials/basic_training 

How to train Diffusion model with additional loss? - Artificial Intelligence Stack Exchange, accessed on April 18, 2025, https://ai.stackexchange.com/questions/40981/how-to-train-diffusion-model-with-additional-loss 

[D] Loss function in Diffusion models : r/MachineLearning - Reddit, accessed on April 18, 2025, https://www.reddit.com/r/MachineLearning/comments/wvnnvb/d_loss_function_in_diffusion_models/ 

Diffusion Models | Towards Data Science, accessed on April 18, 2025, https://towardsdatascience.com/diffusion-models-91b75430ec2/ 

[D] Weird loss behaviour with difusion models. : r/MachineLearning - Reddit, accessed on April 18, 2025, https://www.reddit.com/r/MachineLearning/comments/14wobo3/d_weird_loss_behaviour_with_difusion_models/ 

Diffusion Models for Non-autoregressive Text Generation: A Survey - IJCAI, accessed on April 18, 2025, https://www.ijcai.org/proceedings/2023/0750.pdf

PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model - NIPS papers, accessed on April 18, 2025, https://papers.nips.cc/paper_files/paper/2023/file/fdba5e0a9b57fce03e89cc0cad0a24e9-Paper-Conference.pdf

Diffusion Models: A Comprehensive Survey of Methods and Applications - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2209.00796v13 

Diffusion models in text generation: a survey - PMC, accessed on April 18, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10909201/ 

Review History for Diffusion models in text generation: a survey [PeerJ], accessed on April 18, 2025, https://peerj.com/articles/cs-1905/reviews/ 

http://arxiv.org , accessed on April 18, 2025, https://arxiv.org/pdf/2305.04044

NeurIPS Poster Fast Sampling via Discrete Non-Markov Diffusion Models with Predetermined Transition Time, accessed on April 18, 2025, https://neurips.cc/virtual/2024/poster/95646 

Energy-Based Diffusion Language Models for Text Generation - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2410.21357v3 

Enable Fast Sampling for Seq2Seq Text Diffusion - ACL Anthology, accessed on April 18, 2025, https://aclanthology.org/2024.findings-emnlp.497.pdf

http://openreview.net , accessed on April 18, 2025, https://openreview.net/pdf?id=3s9IrEsjLyk

Empowering Diffusion Models on the Embedding Space for Text Generation - ACL Anthology, accessed on April 18, 2025, https://aclanthology.org/2024.naacl-long.261.pdf

arXiv:2305.14671v2 [cs.CL] 14 Jun 2023, accessed on April 18, 2025, https://arxiv.org/pdf/2305.14671

DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models - ACL Anthology, accessed on April 18, 2025, https://aclanthology.org/2023.findings-emnlp.660.pdf

Chain-of-Thought Reasoning in Diffusion Language Models - arXiv, accessed on April 18, 2025, https://arxiv.org/pdf/2402.07754

Chain-of-Thought Reasoning in Diffusion Language Models - NeurIPS 2025, accessed on April 18, 2025, https://nips.cc/virtual/2024/poster/95935 

Diffusion Language Model with Query-Document Relevance for Query-Focused Summarization - ACL Anthology, accessed on April 18, 2025, https://aclanthology.org/2023.findings-emnlp.735.pdf

SCALING DIFFUSION LANGUAGE MODELS VIA ADAPTATION FROM AUTOREGRESSIVE MODELS - OpenReview, accessed on April 18, 2025, https://openreview.net/pdf?id=j1tSLYKwg8

Chain-of-Thought Reasoning in Diffusion Language Models - NeurIPS 2025, accessed on April 18, 2025, https://neurips.cc/virtual/2024/poster/95935 

kuleshov-group/awesome-discrete-diffusion-models - GitHub, accessed on April 18, 2025, https://github.com/kuleshov-group/awesome-discrete-diffusion-models 

Meta-Diffu$B$: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration | OpenReview, accessed on April 18, 2025, https://openreview.net/forum?id=NTWXVvIXJM 

Likelihood-Based Diffusion Language Models - OpenReview, accessed on April 18, 2025, https://openreview.net/pdf?id=e2MCL6hObn

Unified Multimodal Discrete Diffusion - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2503.20853v1 

arXiv:2410.14157v3 [cs.CL] 18 Feb 2025, accessed on April 18, 2025, https://arxiv.org/pdf/2410.14157

NeurIPS Poster Discrete-state Continuous-time Diffusion for Graph Generation, accessed on April 18, 2025, https://neurips.cc/virtual/2024/poster/94674 

NeurIPS Poster Discrete Flow Matching, accessed on April 18, 2025, https://neurips.cc/virtual/2024/poster/95902 

Structured Denoising Diffusion Models in Discrete State-Spaces - OpenReview, accessed on April 18, 2025, https://openreview.net/forum?id=h7-XixPCAL 

Simple and Effective Masked Diffusion Language Models - NIPS papers, accessed on April 18, 2025, https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf

Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2410.14157v3 

Simplified and Generalized Masked Diffusion for Discrete Data - arXiv, accessed on April 18, 2025, https://arxiv.org/pdf/2406.04329

Simple and Effective Masked Diffusion Language Models - OpenReview, accessed on April 18, 2025, https://openreview.net/forum?id=L4uaAR4ArM&referrer=%5Bthe%20profile%20of%20Volodymyr%20Kuleshov%5D(%2Fprofile%3Fid%3D~Volodymyr_Kuleshov1)

Scaling up Masked Diffusion Models on Text - OpenReview, accessed on April 18, 2025, https://openreview.net/forum?id=WNvvwK0tut 

Conditional [MASK] Discrete Diffusion Language Model - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2411.06438v3 

DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models, accessed on April 18, 2025, https://openreview.net/forum?id=fRmfDqZ2yq 

Discrete-state Continuous-time Diffusion for Graph Generation - NIPS papers, accessed on April 18, 2025, https://proceedings.neurips.cc/paper_files/paper/2024/file/91813e5ddd9658b99be4c532e274b49c-Paper-Conference.pdf

PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model, accessed on April 18, 2025, https://proceedings.neurips.cc/paper_files/paper/2023/file/fdba5e0a9b57fce03e89cc0cad0a24e9-Paper-Conference.pdf

Meta-DiffuB: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration - NIPS papers, accessed on April 18, 2025, https://papers.nips.cc/paper_files/paper/2024/file/91d193b65d0b120d29503590827de1ea-Paper-Conference.pdf

Do diffusion models take a long time to train? - AI Stack Exchange, accessed on April 18, 2025, https://ai.stackexchange.com/questions/43012/do-diffusion-models-take-a-long-time-to-train 

GUIDE: Guidance-based Incremental Learning with Diffusion Models - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2403.03938v1 

CEV-LM: Controlled Edit Vector Language Model for Shaping Natural Language Generations - ACL Anthology, accessed on April 18, 2025, https://aclanthology.org/2024.eacl-long.80.pdf

Daily Papers - Hugging Face, accessed on April 18, 2025, https://huggingface.co/papers?q=LM 

Intelligent Artistic Typography: A Comprehensive Review of Artistic Text Design and Generation - arXiv, accessed on April 18, 2025, https://arxiv.org/html/2407.14774v1

