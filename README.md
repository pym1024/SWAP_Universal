# Zero-Shot Neural Network Evaluation with Sample-Wise Activation Patterns <br/>
Zero-shot proxies, also known as training-free metrics, are widely adopted to reduce the computational overhead in neural network evaluation. Existing zero-shot metrics have several limitations, including weak correlation with the true performance and poor generalisation across different networks or downstream tasks. For example, most of these metrics apply only to either convolutional neural networks (CNNs) or Transformers, but not both. To address these limitations, we propose Sample-Wise Activation Patterns (SWAP), and its derivative, SWAP-Score, a novel, universal, and highly effective zero-shot metric. This metric measures the expressivity of neural networks over a mini-batch of samples, showing a high correlation with the neural networks' ground-truth performance. For both CNNs and Transformers, the SWAP-Score outperforms existing zero-shot metrics across computer vision and natural language processing tasks. For instance, Spearman's correlation coefficient between the SWAP-Score and CIFAR-10 validation accuracy for DARTS CNNs is 0.93, and 0.71 for FlexiBERT Transformers on GLUE tasks. Moreover, SWAP-Score is label-independent, hence can be applied at the pre-training stage of language models to estimate their performance for downstream tasks.

# Usage

The following instruction demonstrates the usage of evaluating network's performance through SWAP-Score.

**/src/metrics/swap.py** contains the core components of SWAP-Score. 

**/datasets/DARTS_archs_CIFAR10.csv** contains 1000 CNNs along with their CIFAR-10 performance (trained for 200 epochs).

**/datasets/BERT_benchmark.json** contains 500 BERT-like Transformers along with their GLUE performance.

* Install necessary dependencies (a new virtual environment is suggested).
```
cd SWAP_Universal
pip install -r requirements.txt
```
* Calculate the correlation between SWAP-Score and ground-truth CIFAR-10 performance of 1000 CNNs.
```
python SWAP_CNN.py
```

* Calculate the correlation between SWAP-Score and ground-truth GLUE performance of 500 BERT-like Transformers.
```
python SWAP_Transformer.py
```
