# AAA534
AAA534-00 컴퓨터비전(영강) 프로젝트

```Bash
# Create a conda environment
conda env create -f tpt.yaml # change prefix in tpt.yaml
conda activate tpt

# Run various experiments (arch, n_ctx, selection_p, tta_steps)
cd TPT
sh scripts/test_tpt_3D.sh # change --output_filepath in the script
```
