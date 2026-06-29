# CelebA Generator Runs

Goal: train DPImageBench generators on CelebA for later membership inference auditing.

Protocol:
- Dataset: CelebA
- Member set: real images used to train the generator
- Nonmember set: real images not used to train the generator
- Current stage: generator training only
- Downstream classifier and MIA are later stages

DPImageBench CelebA split note:
- Official CelebA split: 162,770 train / 19,867 val / 19,962 test
- DPImageBench split may further split official train into train/reference subsets
- Need to record exact train split used by each generator

| Generator | Config | Epsilon | Delta | Train split | Output path | Slurm script | Job ID | Status | Notes |
|---|---|---:|---:|---|---|---|---|---|---|
| DP-MERF | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DP-NTK | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DP-Kernel | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| PE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| GS-WGAN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DPGAN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DPDM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DP-FETA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| PDP-Diffusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DP-LDM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| DP-LoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
| PrivImage | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending | |
