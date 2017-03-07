# randombp
Demo: Feedback Alignment for a Slightly More Bio-Plausible Backprop

This repo contains short demos based on the following papers:

* Random feedback weights support learning in deep neural networks
* Direct Feedback Alignment Provides Learning in Deep Neural Networks

They explore the idea of unpairing the forwards-pass weights of a neural network
and the backwards gradients used for training. Instead, a different,
random matrix is used for the backwards pass.
