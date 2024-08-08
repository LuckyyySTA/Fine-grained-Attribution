# Fine-grained-Attribution

Source code for the ACL 2024 Findings paper "Learning Fine-Grained Grounded Citations for Attributed Large Language Models" (More details will be updated soon)
## Requirements
The required Python packages are listed in [requirements.txt](https://github.com/LuckyyySTA/Fine-grained-Attribution/blob/main/requirements.txt). You can create a new conda environment, then run the following command to install them.

```shell
conda create -n front python=3.10
conda activate front
pip install -r requirements.txt
```

## Data
You can directly download both the raw and processed dataset from this Google Drive [link](https://drive.google.com/drive/folders/1FYrmf2i0rpYcKxluA25Mw48yz1ufsHAI?usp=drive_link).

## Training
We use 4xA100 80G GPUs for the two-stage training.
### Stage1: Grounding Guided Generation

```
cd training/stage1_grounding_guided_generation
sh train_sft.sh
```
### Stage2: Consistency-Aware Alignment

```
cd training/stage2_consistency_aware_alignment
sh train_dpo.sh
```

## Evaluation
For evaluation, please refer to [ALCE](https://github.com/princeton-nlp/ALCE).

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email [lhuang@ir.hit.edu.cn](lhuang@ir.hit.edu.cn)
