# cw_diffusion

This directory keeps the original CellWhisperer evaluation idea, but adapts it for scMMaDA diffusion models.

What changed:
- No LLaVA / `<image>` interface.
- No autoregressive perplexity scoring.
- We score each gold answer with the same masked denoising objective used in MMUG training.
- We compare `matched` vs `mismatched` cell-feature conditioning.

Files:
- `diffusion_evaluation_score.py`: main diffusion evaluator.
- `build_tabsap_celltype_evaluation_dataset.py`: builds cell-type conversations without `<image>` tokens.
- Original notebooks remain for reference, but they are AR/LLaVA-specific and should not be used directly for scMMaDA.

Example:
```bash
  python cw_diffusion/diffusion_evaluation_score.py \
    --train-config configs/scmmada_stage1_ours.yaml \
    --model-path /mnt/c20250607/user/wanghaoran/zyh/scMMaDA/scmmada-stage1-ours-cellfeat-pretrain-old-0/checkpoint-200/unwrapped_model \
    --evaluation-dataset /mnt/c20250607/user/wanghaoran/zyh/datasets/sft_conversations/finetune_conversations.json \
    --feature-root /mnt/c20250607/user/wanghaoran/zyh/datasets/features \
    --output-csv cw_diffusion/results/diffusion_eval.csv \
    --batch-size 8 \
    --num-negatives 10 \
    --score-repeats 4
```

Output compatibility:
- The CSV includes `eval_all_perplexities` for compatibility with the old plotting notebooks.
- In this diffusion setting, that column is not AR perplexity; it is the diffusion denoising score (lower is better).
