

# EVLncRNA-net Trained Models

Due to GitHub's file size limitations, the trained models for EVLncRNA-net can be accessed at the following link:

[**Google Drive: EVLncRNA-net Trained Models**](https://drive.google.com/drive/folders/1AMV-lyQ5ZyLCmlIyKgFl4C0z0wqtp9-S?usp=sharing)

Please download the models from the provided Google Drive folder and place the `model` folder into your code directory.


# Requirements

```plaintext
Bio==1.5.8
matplotlib==3.1.2
numpy==1.23.5
pandas==1.5.3
scikit_learn==1.2.2
torch==2.0.0
torchvision==0.15.0
tqdm==4.65.0
torch==2.1.2+cu121
torch-cluster==1.6.3+pt21cu121
torch_geometric==2.5.3
```

# Usage

### Select the Species You Want to Train

In `config.py`, change the dataset by modifying the `self.data_name` parameter:

```python
self.data_name = 'plant'
```

You only need to adjust this parameter, with options including `human`, `mouse`, and `plant`.

### Select Your Training and Test Sets

Specify the paths for the training and test sets in `config.py`.

### Train Your Model

Run `train.py`:

```python
data = Biodata(fasta_file=config.train_x, label_file=config.train_label,
               feature_file=config.train_fea)
dataset = data.encode(thread=20)
```

This handles dataset processing.

```python
if config.data_name == 'human':
    model = mynet(label_num=2, K=3, d=3, node_hidden_dim=3, other_feature_dim=128, other_feature_dim_in=2000, pnode_nn=False, fnode_nn=False).to(device)  # human_net
else:
    model = mynet(label_num=2, K=3, d=3, node_hidden_dim=3, other_feature_dim=128, other_feature_dim_in=4000, data_name=config.data_name).to(device)  # mouse & plant net
```

This sets up the model architecture.

```python
train(dataset, model, weighted_sampling=False, batch_size=config.batch_size,
      learning_rate=config.learning_rate, epoch_n=20, iffloss=1)
```

This initiates the training process.

Where:

- `iffloss=1` uses FocalLoss.
- Otherwise, CrossEntropyLoss is used as the loss function.

## Use Pretrained Models

After downloading the pretrained models from the `model` folder, select the desired species model in `config.py` and run `Test.py` to generate predictions.



# Requirments

---

Let me know if you need further adjustments!
