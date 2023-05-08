## CS 598 Deep Learning for Healthcare:

Reproducibility Project: Pre-training of Graph Augmented Transformers for Medication Recommendation.

## Project Team: 

Prateek Dhiman : pdhiman2@illinois.edu

Unnati Hasija : uhasija2@illinois.edu

## Introduction

The aim of this project is to understand, replicate and extend the paper Pre-training of Graph Augmented Transformers for Medication Recommendation.
**G-Bert** combines Graph Neural Networks and the well known pre-trained model: BERT (Bidirectional Encoder Representations from Transformers) for 
medical code representation and medication recommendation. Graph neural networks (GNNs) are used to represent the structure information of medical 
codes from a medical ontology. Then this GNN representation is integrated to a transformer-based visit encoder and pre-train it on single-visit EHR 
data. For our project, we reproduced this G-Bert, it's original ablations and our new ablations. This repository contains a modified version of the orginal work, done in order to carry out and extend the experiments.

As an extension to G-Bert and for our feasibility study, we propose a new model: **GGPT** which combines GNN with GPT2 model. We believe, using a Graph Neural Network (GNN) with GPT-2 for medical recommendations could also be a promising approach. GPT-2 is a powerful language model that can generate coherent and fluent text based on the context provided and GNNs can capture the relationships between different medical concepts and entities, and leverage this information for better recommendation generation.

## Feasibility Study: GGPT

GPT-2 model is trained on a larger corpus of general text data, including a broad range of medical knowledge. The idea to train GPT-2 model also on EHRDataset for a single visit and may prove to be a promising approach for medical recommendations. GPT-2 has a larger model size and higher number of parameters: 1.5 billion, which may allow it to capture more complex relationships between medical concepts and generate accurate recommendations. GPT-2 generates fluent and coherent text due to its architecture, which includes an autoregressive language modeling component. This could be advantageous for generating natural-sounding medical recommendations that are more likely to be understood by patients and healthcare professionals.



## Steps to replicate/Reproducibility:

## 1. Dependencies:

To reproduce the results or to be able to execute the G-Bert code, we were missing some libraries. We created a list of those libraries in requirements.txt.
You may install it using:
$ pip install -r requirements.txt

## 2. Training code:

The dataset was already split into training, validation and testing set in ratio of 0.6,0.2,0.2 respectively. The split is done in EDA.ipynb file train-id.txt, eval-id.txt, test-id.txt and the files are stored accordingly.
The script for training the G-Bert model is provided in run_alternative.sh bash script. The script executes the the python file run_pretraining.py to pretrain the G-Bert model on EHR Dataset. Then it executes the G-Bert prediction on this pre-trained model. The script alternates the pre-training with 5 epochs and fine-tuning procedure with 5 epochs for 15 times to stabilize the training procedure.
For our project, we used the same procedure of executing pre-training with and without graphs and with and without pre-training on EHR Dataset.
We also executed the script after changing the Graph model from GAT to GCN(Graph Convolution Network) and GTN(Graph Transform Network) to test our ablations.

## 3. Evaluation code 

We evaluated the original results by executing:
python run_gbert.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-predict --graph

**To test our main ablations:**
1. Clone the git repository.
2. Install the requirements: pip install -r requirements.txt
3. Comment the code for GATConv in graph_models.py and uncomment the code for GTNConv or GCNConv.
4. Execute the corresponding command:

For GTNConv: python run_gbert.py --model_name GBert-predict-qGTN1 --use_pretrain --pretrain_dir ../saved/GBert-predict-qGTN1 --graph

For GCNConv: python run_gbert.py --model_name GBert-predict-qGCN --use_pretrain --pretrain_dir ../saved/GBert-predict-qGCN --graph

The pre-trained models based on GTN and GCN are placed in the GitHub repository.

**To test our feasibility study approach:**
1. After installing the requirements, execute the Jupyter Notebook: **GGPT2.ipynb**. The pre-trained GPT2 model is a part of this GitHub repository.

Please Note: Information on other ablations can be found in the paper
## 4. Pre-training and pre-trained models:

In the run_pretraining.py, BERT model is pre-trained on the EHRDataset (both single-visit EHR sequences and multi-visit EHR sequences). 
In here, the 15% of the tokens are replaced by [MASK] and [CLS] is the first token of each sentence. The pre-training code creates a model with the config specified in config.py. 
To train the model with our ablations in place, we added the code in graph_model.py and used the same steps to pre-train using run_pretraining.py and later testing using run_gbert.py. We used the train mechanism as used by the authors. 

## Baselines and ablations:

To prove the claims and ablations in the original paper and to test our ablations, we took the baselines as follows:
We started with using local CPU based machines(laptops) and later switched to Azure GPU VM: - 1 x NVIDIA Tesla K80: Standard_NC6 (6 cores, 56 GB RAM, 380 GB disk).

## Baselines:
To prove the claims in the paper:
1. G-Bert performs better with graphs and pre-training 
2. G-Bert performs better than GAMENet and RETAIN.

we ran the following baselines:

|    Model        |    F1 score   |    PR AUC     |  Jaccard Score |
|-----------------|-------------- |---------------|----------------|
| G-Bert(Original Paper)|0.6152|	  0.6960      |    0.4565      |
| G-Bert(Our replication)	      |       0.6065  |	  0.6906      |    0.4478      |
| G-Bert G-	      |       0.6038  |	  0.6906      |    0.4478      |
| G-Bert G-P-     |       0.5528  |	  0.6366      |    0.3908      |
| G-Bert P-	      |       0.5351  |	  0.6206      |    0.3726      |
| GAMENet	      |       0.3497  |	  0.4561      |    0.2320      |
| GAMENet D-      |       0.3596  |	  0.4326      |    0.2399      |
| RETAIN 	      |       0.3127  |	  0.4606      |    0.2013      |
   		

## Ablations: 
We attempted the below ablations and after changing the GATConv to GTNConv, the PR Accuracy was better than the original G-Bert. 
  
|    Model        |    F1 score   |    PR AUC     |  Jaccard Score |
|-----------------|-------------- |---------------|----------------|
|Aggr_out from sum to mean| 0.5710 |0.6554|0.4096|
|Aggr_out from sum to max|0.5550|0.6318|0.3942|
|Attention heads (4->8)|0.5245|0.6201|0.3644|
|GATConv to GCNConv|0.5438|0.6296|0.3824|
|GATConv to GTNLayer Attn heads = 4|0.6063|0.6896|0.4463|
|**GATConv to GTNLayer Attn heads = 6**|**0.6121**|**0.6967**|**0.4534**|
|Leaky_relu->sigmoid|0.5125|0.5464|0.3533|
|Softmax->Sigmoid|0.4808|0.5810|0.3253|
|Leaky_relu->tanh(Mish)|0.5904|0.6722|0.4298|

For more detailed information on the hyperparameter settings and the training process, please refer to the UIUC_DLH_Spring23.pdf file included in the repository. This report provides a comprehensive overview of our experimental setup and analysis of the results.

# Citation
@article{shang2019pre,
  title={Pre-training of Graph Augmented Transformers for Medication Recommendation},
  author={Shang, Junyuan and Ma, Tengfei and Xiao, Cao and Sun, Jimeng},
  journal={arXiv preprint arXiv:1906.00346},
  year={2019}
}
