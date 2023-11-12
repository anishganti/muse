# Open Source Muse
We are a part of Roundtable ML's text-to-image group that's working on an open source implementation of the Muse model from Google.
## Instructions
### Installing The Requirements
After git cloning the repository, make sure to run the following script.
```
pip install -r requirements.txt
```
### Installing Dependencies
Our model also depends on pretrained VQGAN models from [Laion](https://github.com/huggingface/open-muse), so clone their repository to get all the dependencies required.
```
git clone https://github.com/huggingface/open-muse
cd open-muse
pip install -e ".[extra]"
```
### Configuring Compute Resources
For the purpose of our project, we used Google Cloud's Virtual Machine services and use the following specifications for our compute resource.

- Machine Type: g4-standard4
- GPU: NVIDIA L4
- Disk Space: 500 GB
- Image: Google, Deep Learning VM for PyTorch 2.0 with CUDA 11.8, M112, Debian 11, Python 3.10, with PyTorch 2.0 and fast.ai preinstalled.

If you have more budget than we do feel free to multiply the number of GPU's for faster computation, but we were limited to the free credits from Google Cloud so we used only 1.

### Training and Model Configurations
We use the following training configurations that allows us to nearly replicate the training process from the paper.
- Batch Size: 16
- Grad Accumulation Size: 32
- Mixed Precision Training: True
- Epochs: 1

If during the training process, the loss becomes NaN after some time, try to turn off Mixed Precision Training by setting `args.amps=False`. If during the process the compute runs out of CUDA memory, half the batch size and double the grad accumulation size. The point is that the product of the grad accumulation and batch sizes should equal to the batch size that is being simulated.

### Running The Training Scripts
To run the training script,
```
torchrun --standalone --nproc-per-node=1 train_base.py
```
This will allow you to train the model using 1 GPU, but you can also increase the argument `nproc-per-node` to train on multiple GPU's. Keep in mind, you need might need to tweak the training script a little bit to make sure the gradients sync across GPU's since each node will have different batches.

### Parallel Decoding
TBA

## Data
The data we decided to use was LAION's 400 Million pairs of text-image pairs. However, we noticed that some links cannot be accessed in the dataset, so we replaced these images with solid pictures of random colors. 

## Pre-trained Models
Due to limitations in our compute resources, we use the 'base' size model of the T5 Encoder which has an embedding dimension of 768. We also used Laion's pretrained VQ models. In particular, we used their VQGAN model for training the base model and the PaellaVQ model for the super-resolution model.

## Architecture
Although the paper gives us an oveview of the architecture, they don't give the fine details which leaves us some room for our own take of the model. We differed from most implementations of the Muse model by implementing our own version of multi-axis attention for the super resolution model and using the output for cross-attention between the flattenend base model output. 

### Multi-Axis Attention
Our implementation sub-divides a $64 \times 64$ image tokens into 64 patches of size $8 \times 8$. We then flatten each patches transforming the dimension from $H \times W\$ to $P \times (H \times W)\$. We project this input and sub divide the embedding dimension into $h$ different heads. Afterwards, we split half of the heads and use the last half to swap the dimensions between the number of patches and patches length. We do this so that our model captures within and across patches attention. The multi-axis attention mechanism is important because it reduces the computation time for self-attention from  $O(n^2)$ to $O(\sqrt(n))$.

### Model Configuration & Size
TBA

## Results
