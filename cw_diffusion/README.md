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
  --model-path /path/to/checkpoint-500/unwrapped_model \
  --evaluation-dataset /path/to/eval_conversations.json \
  --feature-root /data/bgi/data/projects/multimodal/zyh/datasets/features \
  --output-csv cw_diffusion/results/diffusion_eval.csv \
  --batch-size 8 \
  --num-negatives 10 \
  --score-repeats 4
```

Output compatibility:
- The CSV includes `eval_all_perplexities` for compatibility with the old plotting notebooks.
- In this diffusion setting, that column is not AR perplexity; it is the diffusion denoising score (lower is better).
