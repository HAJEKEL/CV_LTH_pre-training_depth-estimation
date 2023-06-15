# Depth estimation fine tuning Reproduction

This repository contains instructions for reproducing the experiments that answer the research question:
Can we aggressively trim down the complexity of pre-trained models, without damaging their downstream transferability on the depth estimation task?
The experimental setup involves a google colab [ipynb file](https://github.com/HAJEKEL/CV_LTH_pre-training_depth-estimation) running on a Google Compute Engine (GCE) virtual machine instance with a GPU. The source code to obtain the masks through iterative magnitude pruning  code can be found in the [CV_LTH_Pre-training](https://github.com/HAJEKEL/CV_LTH_Pre-training) repo. 

## Prerequisites

Before starting the fine tune process for the depth estimation task on the pruned and non-pruned task-agnostic pre-trained model, make sure you have the following:

- A Google Cloud Platform (GCP) account.
- A Google Compute Engine (GCE) project with GPU quota in all regions.
- Basic knowledge of using GCE instances and connecting to them from a Jupyter notebook.


## Reproduction Steps

To reproduce the fine tuning process, follow the steps below:

1. Create a Google Compute Engine virtual machine instance with at least one GPU in all regions. You can follow the steps below:
   - Go to the [GCP Console](https://console.cloud.google.com/) and create a new project if you haven't done so already.
   - Go to the **IAM & admin > Quotas** page and search for "gpus_all_regions".
   - Select the "GPUs (all regions)" quota and click **Edit Quotas**. Apply for at least 1 GPU in all regions. You can use **fast.ai** as description for the quota application.
   - Wait untill you get a conformation in you mailbox. 
2. Launch a Google Colab virtual machine instance with a GPU:
   - Search for "Colab Marketplace" and click on the first result.
   - Deploy the VM with your desired configuration, the standard given one is fine. 
   - If there are no available VM instances in your current region, iterate through different zones until you find one with available instances.
3. Connect your Google Colab runtime to the GCE instance:
   - Copy the [ipynb file](https://github.com/HAJEKEL/CV_LTH_pre-training_depth-estimation/blob/main/fine_tuning.ipynb) in google drive. Open it with google colab, click on "Connect to a custom GCE VM".
   - Add the GCE project ID, zone of the VM, and the VM instance name.
4. Follow the steps inside the [ipynb file](https://github.com/HAJEKEL/CV_LTH_pre-training_depth-estimation/blob/main/fine_tuning.ipynb):
   - Comments inside the file explain what's going on. 
   - Note that the source code allows for fine tuning on 3 different datasets, ImageNet, MoCo and SimCLR. However, at the moment, only the masks of ImageNet are available. Therefore, for now, the only experiment that can be conducted is based on the ImageNet dataset.

## Pre-trained Resnet-50 on Imagenet

The source code for pre-training the Resnet-50 on imagenet  can be found in the [CV_LTH_Pre-training](https://github.com/HAJEKEL/CV_LTH_Pre-training) repo. The README there will guide you through the source code. 