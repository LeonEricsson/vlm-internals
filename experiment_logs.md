### Investigating why spatial reasoning is hard for VLMs

There was a paper released just a few months back "Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas" which caught my eye. It was the reason for creating this repo in the first place as I wanted
to play around with the findings. The analysis presented in the beginning 4 sections of this paper build on an analysis of the attention distribution. To summarize briefly

the paper investigates VLMs on spatial reasoning tasks (e.g., prompt: <image_pad> ... Is the cat under the bed? -> model outputs "True" or "False"). A key finding they report: "the model places 90% of its attention on text tokens from the final predicting token's position, even when text tokens only make up 10% of the total sequence."

From what I can gather from their methods/code, they arrive at this by:
Identifying the final token position (i.e., the one immediately preceding/generating the "True" or "False" output).
Analyzing the attention distribution from this specific final token position, looking at what it attends to across the input sequence, across the model layers.

I was keen on performing the same analysis for newever models, because the paper was looking at Llava 1.5/1.6 which by todays standards are quite outdated. Specifically I wanted to look at how attention looks inside Qwen 2.5 VL.

I built most of the necessary tooling to extract and perform the analysis on the VSR benchmark dataset.

It wasn't untill I started analyzing the results until I started questioning what I was doing. Can we really attribute attention scores in layers to the original symbol. Naturally, after thinking about it, I landed in this being quite unreasonable.

what we're doing here is blindly assuming that the attended channels actually carry the spatial features we are interested in. We are assuming that the spatial features are also directly passed from the image token into the final output token. This is wrong

confirm whether the attended channels actually carry spatial features; in doing so, they inherit the long-standing pitfall of equating softmax weights with information flow. By pooling attention across heads and layers they reduce a rich, circuit-level mechanism to a single saliency heat-map, masking the head-specific routes that prior mechanistic-interpretability work has shown to mediate geometry reasoning.

another problem is we are assuming that the image patch embeddings passed to us by the ViT, have not already entangled a bunch of spatial information between the patches.

#### Alternative ways to analyse attention:

Attention flow / attention rollout
https://aclanthology.org/2020.acl-main.385.pdf

https://jacobgil.github.io/deeplearning/vision-transformer-explainability

### Paths forward

Find where in the network the spatial relationship is embedded. Probe through linear probes. Go through the activations of all layers and see at which point we can no longer find the spatial relationship in the residual stream.
