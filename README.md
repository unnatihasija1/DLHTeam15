## Final Project: 

Reproducibility Project: Pre-training of Graph Augmented Transformers for Medication Recommendation

## Project Team: 

Prateek Dhiman : pdhiman2@illinois.edu

Unnati Hasija : uhasija2@illinois.edu

## Introduction

G-Bert combines Graph Neural Networks and thr well known pre-trained model: BERT (Bidirectional Encoder Representations from Transformers) for 
medical code representation and medication recommendation. Graph neural networks (GNNs) are used to represent the structure information of medical 
codes from a medical ontology. Then this GNN representation is integrated to a transformer-based visit encoder and pre-train it on single-visit EHR 
data. For our project, we reproduced this G-Bert, it's original ablations and our new ablations. 

As an extension to G-Bert and for our feasibility study, we propose a new model: GGPT2 which combines GNN with GPT2 model. We believe, using a Graph Neural Network (GNN) with GPT-2 for medical recommendations could also be a promising approach. GPT-2 is a powerful language model that can generate coherent and fluent text based on the context provided and GNNs can capture the relationships between different medical concepts and entities, and leverage this information for better recommendation generation.


## Steps to replicate/Reproducibility:

## 1. Dependencies:

To reproduce the results or to be able to execute the G-Bert code, we were missing some libraries. We created a list of those libraries in requirements.txt.
You may install it using:
$ pip install -r requirements.txt

## 2. Training code:

The dataset is split into training, validation and testing set in ratio of 0.6,0.2,0.2 respectively. The split is done in EDA.ipynb file train-id.txt, eval-id.txt, test-id.txt 
and the files are stored accordingly.
The script for training the G-Bert model is provided in run_alternative.sh bash script. The script basically executes the pretraining.py to pretrain the G-Bert on EHR Data.
Then it executes the G-Bert prediction on this pre-trained model. The script alternates the pre-training with 5 epochs and fine-tuning procedure with 5 epochs for 15 times to stabilize the training procedure.
For our project, we adjusted the above script to execute pre-training with and without graphs and with and without pre-training.
We also executed the script after changing the GAT model to GCN and GTN to test our ablations.

## 3. Evaluation code 

We evaluated the results by executing:
python run_gbert.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-predict --graph

## 4. Pre-training and pre-trained models:

In the run_pretraining.py, BERT model pre-trained on the EHRDataset (both single-visit EHR sequences and multi-visit EHR sequences). 
In here, the 15% of the tokens are replaced by [MASK] and [CLS] is the first token of each sentence. 
The pre-training code creates a model with the config specified in config.py.

## Baselines and ablations:

To prove the claims and ablations in the original paper and to test our ablations, we took the baselines as follows:
We used local CPU and on GPU - 1 x NVIDIA Tesla K80: Standard_NC6 (6 cores, 56 GB RAM, 380 GB disk).

## Baselines:
   Model	        F1 score	  PR-AUC	Jaccard Score
G-Bert 	            0.6065 	  0.6906         0.4478   
G-Bert G-	         0.6038	  0.6824	        0.4452    
G-Bert G- P-			
G-Bert P-			
GAMENet			
GAMENet D-			

## Ablations: 
  
<insert the table here>

## New Approach: GGPT

Since GPT-2 is trained on a large corpus of general text data, which also includes a broad range of medical knowledge. The idea is, GPT-2 also to be trained on 
EHRDataset for single visit and may prove to be a promising approach for medicial recommendations. GPT-2 has a large model size and high number of parameters: 1.5 billion, which may allow it to capture more complex relationships between medical concepts and generate accurate recommendations. GPT-2 generates fluent and coherent text due to its architecture, which includes an autoregressive language modeling component. This could be advantageous for generating natural-sounding medical recommendations that are more likely to be understood by patients and healthcare professionals.
   

# Citation
@article{shang2019pre,
  title={Pre-training of Graph Augmented Transformers for Medication Recommendation},
  author={Shang, Junyuan and Ma, Tengfei and Xiao, Cao and Sun, Jimeng},
  journal={arXiv preprint arXiv:1906.00346},
  year={2019}
}
