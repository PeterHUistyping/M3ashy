The weights of the pre-trained base models are uploaded at Hugging Face [Synthesis model weights](https://huggingface.co/Peter2023HuggingFace/M3ashy). Please download the model weights and put them in the `data/NeuMERL` folder.

There are two options to use the NeuMERL dataset,

Options 1: 2400 materials in a single file (default).

    NeuMERL-2400.npy

Options 2: 100 materials per file, with a total of 24 files.

    NeuMERL(24*100)/mlp_weights_all_{i}.npy

    For option 2, please update NBDRFWeightsDataset parameters in`src/pytorch/dataset/nbrdf_weights_dataset.py`

    ``   load_all_files=False,     max_files=25  # any value from 2 to 25.   ``
