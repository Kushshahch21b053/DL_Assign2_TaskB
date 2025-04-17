# DL Assign2 Part B

### Github Link for Part A

https://github.com/Kushshahch21b053/DL_Assign2_TaskB

### Wandb Report Link (Part A and B)

https://api.wandb.ai/links/ch21b053-indian-institute-of-technology-madras/usnef6k8

### Code organisation for Part B

- config.py
- dataset.py
- model_finetune.py
- train.py
- evaluate.py
- visualize.py
- main.py
- requirements.txt

### How to run code

- Firstly, if needed, you can do:
```
pip install -r requirements.txt
```

**Task B Question 1-3**

- The code can be run using the following:
```
python main.py \
  --data_dir (your dataset path) \
  --fine_tune_blocks 2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --epochs 10 \
  --weight_decay 0.001 \
  --random_seed 42 \
  --plot_grid
```
