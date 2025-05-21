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

