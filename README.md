# NeuMaDiff: Neural Material Synthesis via Hyperdiffusion

<p align="center"><a href="https://chenliang-zhou.github.io">Chenliang Zhou</a>, <a href="https://peterhuistyping.github.io/">Zheyuan Hu</a>, <a href="https://asztr.github.io/">Alejandro Sztrajman</a>, <a href="https://caiyancheng.github.io/academic.html">Yancheng Cai</a>, <a href="https://www.cst.cam.ac.uk/people/yl962">Yaru Liu</a>, <a href="https://www.cl.cam.ac.uk/~aco41/">Cengiz Öztireli</a></p>

<p align="center">Department of Computer Science and Technology<br>University of Cambridge</p>

<p align="center">
    <a href="https:/peterhuistyping.github.io/NeuMaDiff/">[Project page]</a>  
    <a href="https://arxiv.org/abs/2411.12015">[Paper]</a>
    <a href="https://huggingface.co/Peter2023HuggingFace/NeuMaDiff">[Base model]</a>
    <a href="https://huggingface.co/datasets/Peter2023HuggingFace/NeuMERL">[NeuMERL dataset]</a>
</p>

![teaser](./docs/img/teaser.png)

3D models and scene rendered with our synthesized neural materials.

# Abstract

High-quality material synthesis is essential for replicating complex surface properties to create realistic digital scenes. However, existing methods often suffer from inefficiencies in time and memory, require domain expertise, or demand extensive training data, with high-dimensional material data further constraining performance. Additionally, most approaches lack multi-modal guidance capabilities and standardized evaluation metrics, limiting control and comparability in synthesis tasks.

To address these limitations, we propose **NeuMaDiff**, a novel **neu**ral **ma**terial synthesis framework utilizing hyper **diff**usion. Our method employs neural fields as a low-dimensional representation and incorporates a multi-modal conditional hyperdiffusion model to learn the distribution over material weights. This enables flexible guidance through inputs such as material type, text descriptions, or reference images, providing greater control over synthesis.

To support future research, we contribute two new material datasets and introduce two BRDF distributional metrics for more rigorous evaluation. We demonstrate the effectiveness of NeuMaDiff through extensive experiments, including a novel statistics-based constrained synthesis approach, which enables the generation of materials of desired categories.

# Dataset and base model

Our NeuMERL dataset weights are upload at AI community Hugging Face [NeuMERL dataset](https://huggingface.co/datasets/Peter2023HuggingFace/NeuMERL).

The checkpoint of the base models are upload at Hugging Face [Synthesis model checkpoint](https://huggingface.co/Peter2023HuggingFace/NeuMaDiff).

# Installation
Environment: TODO 

# How to run

TODO

TODO: also add a Python notebook

# Citation

If you found the paper or code useful, please consider citing,

```
@misc{
       NeuMaDiff2024,
      title={NeuMaDiff: Neural Material Synthesis via Hyperdiffusion}, 
      author={Chenliang Zhou and Zheyuan Hu and Alejandro Sztrajman and Yancheng Cai and Yaru Liu and Cengiz Oztireli},
      year={2024},
      eprint={2411.12015},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2411.12015}, 
}
```
