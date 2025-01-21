# AlphaRetro: Evolutionary Retrosynthesis Planning

The following code is executed in Linux system.

## Quickstart
```bash
git clone https://github.com/ilog-ecnu/AlphaRetro
cd AlphaRetro
```

## Single-step installation
```bash
conda env create -f env_single.yml
conda activate alpha_retro
```

## Multi-step installation
```bash
conda env create -f env_multi.yml
conda activate single_step
```

## Data and model preparation
USPTO_50K: [Google Drive USPTO_50K](https://drive.google.com/drive/folders/1T57KdtR3Ti2I7Ldl3OXBEoN-2HF9G7wf?usp=sharing)

Pistachio: [Nextmove Pistachio](https://www.nextmovesoftware.com/pistachio.html)

Building block dataset: [Enamine Building Block](https://enamine.net/building-blocks)

All pre-trained model can be download from [Google Drive model](https://drive.google.com/drive/folders/1TQ9rCcK9WImPxO3_yr5U8Z712IHmT1wG?usp=sharing)

## Single-step training
Prepare the dataset in the format under `data/t5_data/50k_example.csv`, and then pass the path to the main function to start the training.
```bash
cd single_step/t5
conda activate single_step
python train.py
```

## Multi-step searching
The single-step model and reaction-type model, after being trained or downloaded, are mounted via the server's `serve.py` file and then accessed through the client.

Then the client can be used to search for the retrosynthesis of the given molecule:
```bash
conda activate alpha_retro
python -u multi-step.py > multi-step.log 2>&1
```

## Citation
If you find this repository helpful, please give it a star.