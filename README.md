# Towards Efficient Target-Level Machine Unlearning Based on Essential Graph

A Python implementation of [Towards Efficient Target-Level Machine Unlearning Based on Essential Graph]

## Abstract:

Machine unlearning is an emerging technology that has come to attract widespread attention. A number of factors, including regulations and laws, privacy, and usability concerns, have resulted in this need to allow a trained model to forget some of its training data. Existing studies of machine unlearning mainly focus on unlearning requests that forget a cluster of instances or all instances from one class. While these approaches are effective in removing instances, they do not scale to scenarios where partial targets within an instance need to be forgotten. For example, one would like to only unlearn a person from all instances that simultaneously contain the person and other targets. Directly migrating instance-level unlearning to target-level unlearning will reduce the performance of the model after the unlearning process, or fail to erase information completely. To address these concerns, we have proposed a more effective and efficient unlearning scheme that focuses on removing partial targets from the model, which we name ``target unlearning". Specifically, we first construct an essential graph data structure to describe the relationships between all important parameters that are selected based on the model explanation method. After that, we simultaneously filter parameters that are also important for the remaining targets and use the pruning-based unlearning method, which is a simple but effective solution to remove information about the target that needs to be forgotten. Experiments with different training models on various datasets demonstrate the effectiveness of the proposed approach.

# How to use

## 1. Create the env

```
conda **env** create -f targetunlearning.yaml
```

## 2. Training the original model

```
python unlearning_multi_celeba_resnet18_original.py
#parser.add_argument('--task', type=list, default=['Bald', 'Mouth_Slightly_Open'], help='Attributes')
```

## 3. Generating the graph with balance

```
python unlearning_multi_celeba_resnet18_generate_graph.py
#parser.add_argument('--task_labels', type=list, default=['Bald','Mouth_Slightly_Open'], help='Attributes')
#parser.add_argument('--target_class', type=int, default=0, help='which image you would like to show')
#parser.add_argument('--proportion', type=float, default=0.1, help='Learning Rate')
```

## 4. Executing the target unlearning

```
python unlearning_multi_celeba_resnet18_unlearning.py
#parser.add_argument('--task', type=list, default=['Bald','Mouth_Slightly_Open'], help='Attributes')
#parser.add_argument('--target_class', type=int, default=0, help='which image you would like to show')
```
