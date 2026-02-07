# Generative Competition
This is the implementation of a framework for studying competition among generative models under heterogeneous user preferences.  

The default setting uses CIFAR-10 and DDPM generators, but the framework is modular and extensible.

## Project Structure
```
Generative_Competition/

├── RealData/                       # Real data experiments
│ ├── configs/                      # Example experiment configurations
│ │   ├── models_list.txt           # Example list of model checkpoints used to compete
│ │   └── users.json                # Example user preference specification
│ ├── core/
│ │ ├── choice.py                   # Core market logic: score construction, best-response, welfare/diversity
│ │ ├── generator.py                # DDPM generator
│ │ ├── plots.py                    # Visualization
│ │ └── reward.py                   # Reward model
│ ├── build_S_from_models.py        # Build user-model score matrix
│ ├── compute_choice_weights.py     # Compute choice-aware class weights
│ ├── run_br_from_S.py              # Run best-response dynamics
│ ├── train_ddpm.py                 # Baseline DDPM training
│ └── train_ddpm_grad.py            # Direct-gradient DDPM training
│
├── Synthetic/                      # Synthetic data experiments
│ ├── models.py                     # Models
│ ├── users.py                      # User preferences
│ ├── br.py                         # Best-response dynamics
│ ├── metrics.py                    # Welfare and diversity metrics
│ ├── routing.py                    # Strategy update logic
│ └── run_sim.py                    # Synthetic experiment entry point
│
└── README.md
```

## Environment Requirements

Recommended environment:
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- torchvision, numpy, pandas, matplotlib

## RealData Pipeline

A typical workflow in `RealData/` is:

**1. Train baseline generators**

- `train_ddpm.py`  
    Train a baseline DDPM on CIFAR-10:
    ```bash
    torchrun --nproc_per_node=4 train_ddpm.py --dataset_root ./data
    ```
    Train Specialized Models via LoRA. Fine-tune from the baseline checkpoint with LoRA on selected CIFAR-10 subsets:
    ```bash
    torchrun --nproc_per_node=4 train_ddpm.py --dataset_root ./data --epochs 50 \
    --resume checkpoints/ddpm_baseline.pt \
    --lora_rank 4 --lora_alpha 16 --lora_scale 1.0 \
    --classes "cat,dog" --save checkpoints/ddpm_lora_catdog.pt
    ```
**2. Evaluate models and build score matrix**
    
- `build_S_from_models.py`  
    Compute the user–model score matrix by sampling from each generator and evaluating rewards:
    ```bash
    python build_S_from_models.py \
    --models_list configs/models_list.txt \
    --users_mix  configs/users.json \
    --n_eval 5000 --timesteps 1000 \
    --hub_entry cifar10_resnet20 \
    --reward_mode probs \
    --outdir outputs/S
    ```

**3. Run market competition (Best-Response dynamics)**
- `run_br_from_S.py`  
    Simulate best-response dynamics, and compute utilities, welfare, and diversity metrics.
    ```bash
    python run_br_from_S.py --S_dir outputs/S --players 3 --rounds 2 --outdir outputs/BR
    ```

    Optional step/round plotting:
    ```bash
    python plot_steps.py --history outputs/BR/history.csv \
    --S_dir outputs/S \
    --out outputs/BR/steps.png \
    --xaxis step --round_ticks --overlay W
    ```

**4. Training Data Resampling**
- `compute_choice_weights.py`  
    Translate competition outcomes into class-level weights or choice signals.
    ```bash
    torchrun --nproc_per_node=4 compute_choice_weights.py \
    --S_dir outputs/S \
    --opponent_cols 0,1,2,3,5 \
    --ckpt checkpoints/ddpm_baseline.pt \
    --users_mix configs/users.json \
    --n_eval 1024 --timesteps 1000 \
    --beta 4 --gamma 1 --adaptive_scale \
    --outdir outputs/BR_datasample/iter0
    ```
   
    Train the model as before:
    ```bash
    torchrun --nproc_per_node=4 train_ddpm.py \
    --dataset_root ./data --epochs 20 --workers 8 \
    --lora_rank 4 --lora_alpha 16 --lora_scale 1.0 \
    --resume checkpoints/ddpm_baseline.pt \
    --class_weights_json   outputs/BR_datasample/iter0/choice_weights.json \
    --save checkpoints/ddpm_iter000.pt
   ```

   Then evaluate models and build score matrix.

   Repeats the same pattern by swapping --ckpt and --outdir/--resume to iter1, iter2....

**5. Gradient-Based Training**
- `train_ddpm_grad.py`  
    Train DDPM models using differentiable choice-based rewards.
    ```bash
    torchrun --nproc_per_node=4 train_ddpm_grad.py \
    --dataset_root ./data --epochs 30 --batch_size 128 \
    --lora_rank 4 --lora_alpha 16 --lora_scale 0.5 \
    --users_mix configs/users.json \
    --S_dir outputs/S --opponent_cols 0,1,2,3,5 \
    --lambda_choice 0.5 \
    --save checkpoints/ddpm_grad.pt
   ```


## Related Codebases

[chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models): pretrained CIFAR-10 classifiers used as reward model.
