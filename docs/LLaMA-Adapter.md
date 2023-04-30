## LLaMA-Adapter

LLaMA-Adapter is an efficient fine-tuning method that adapts LLaMA into a well-performed instruction-following model. It demonstrates superior resource efficiency to Alpaca and can be extended to multi-modal input, such as images, for image-conditioned LLaMA. The method involves appending a set of learnable adaption prompts as a prefix to the input instruction tokens in LLaMA's higher transformer layers. It adopts a zero-init attention mechanism with zero gating to adaptively inject new instructional cues into LLaMA while preserving its pre-trained knowledge.

### Key Characteristics

1. 1.2M learnable parameters
2. One-hour fine-tuning
3. Plug with expertise
4. Multi-modal condition support

### Applications

LLaMA-Adapter can be used in various natural language processing tasks that require instruction-following models. Its ability to adapt to multi-modal input also makes it suitable for tasks involving image-conditioned LLaMA.
## Image-Conditioned LLaMA

LLaMA-Adapter can be extended to multi-modal input, such as images, for image-conditioned LLaMA, which achieves superior reasoning capacity on ScienceQA. The method adopts a zero-init attention mechanism with zero gating, which adaptively injects the new instructional cues into LLaMA while effectively preserving its pre-trained knowledge. This approach can potentially be further integrated with wider multi-modal inputs, such as audio and video, and tested on larger LLaMA models and diverse benchmarks in the future.

## Comparison Table: LLaMA-Adapter vs. Alpaca

| Feature                   | LLaMA-Adapter                                      | Alpaca                                           |
|---------------------------|----------------------------------------------------|--------------------------------------------------|
| Resource Efficiency       | 1.2M learnable parameters, 1-hour fine-tuning      | Less resource-efficient, slower fine-tuning      |
| Multi-modal Input Support | Image-conditioned LLaMA, potential for audio/video | Not mentioned                                    |
| Performance on ScienceQA  | Superior reasoning capacity                       | Not mentioned                                    |
| Method                    | Zero-init attention, zero gating                   | Different method                                 |
