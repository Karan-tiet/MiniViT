# Vision Transformer Implementation 

This is a simple PyTorch implementation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). The goal of this project is to provide a simple and easy-to-understand implementation. The code is not optimized for speed and is not intended to be used for production.

## Usage
Dependencies:
- PyTorch 1.13.1 ([install instructions](https://pytorch.org/get-started/locally/))
- torchvision 0.14.1 ([install instructions](https://pytorch.org/get-started/locally/))
- matplotlib 3.7.1 to generate plots for model inspection

Run the below script to install the dependencies
```bash
pip install -r requirements.txt
```

You can find the implementation in the `vit.py` file. The main class is `ViTForImageClassification`, which contains the embedding layer, the transformer encoder, and the classification head. All of the modules are heavily commented to make it easier to understand.

The model config is defined as a python dictionary in `train.py`, you can experiment with different hyperparameters there. 

The model was trained on the CIFAR-10 dataset for 100 epochs with a batch size of 256. The learning rate was set to 0.01 and no learning rate schedule was used. The model config was used to train the model:

```python
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
```

The model is much smaller than the original ViT models from the paper (which has at least 12 layers and hidden size of 768) as I just want to illustrate how the model works rather than achieving state-of-the-art performance.

These are some results of the model:

![](/assets/metrics.png)
*Train loss, test loss and accuracy of the model during training.*

The model was able to achieve 75.5% accuracy on the test set after 100 epochs of training.

![](/assets/attention.png)
*Attention maps of the model for different test images*

You can see that the model's attentions are able to capture the objects from different classes pretty well. It learned to focus on the objects and ignore the background.

These visualizations are generated using the notebook `inspect.ipynb`.

