# Probabilistic inference in networks

## Description
Networks are a powerful tool to represent datasets with pairwise interactions between individual units.  Relevant examples are social networks, representing interactions between people, biological datasets as protein-protein interactions or gene-disease associations; financial transactions between institutions or companies.  
In this course, we will learn how to perform inference tasks in networked datasets, where the minimal input information is a list of edges.   

We will cover:  
- how to learn hidden patterns like community structure (or node clustering) or hidden hierarchies of nodes.  

We will use:  
- probabilistic methods
- inference techniques  (e.g. expectation-maximization and variational inference). 

## Learning Objectives.
By the end of this course, you should be able to:
1. LO1: **Analyze** a networked dataset using probabilistic modeling
2. LO2: **Anfer** hidden patterns like clusters or rankings using statistical inference techniques for networks
3. LO3: **Develop** a basic Phyton program to analyze a network dataset 
4. LO4: **Evaluate** performance of different methods in inference tasks on networks 

### Prerequisites.
   - Basic probability theory and linear algebra. 


## Plan
 
1. [L1: Stochastic block model](https://github.com/cdebacco/ds_prob_inf_net/blob/main/lectures/L1/L1.pdf)
   - LO1.1: **Analyze** a networked dataset using the stochastic block model
   - LO1.2: **Infer** hidden clusters or communities from a network dataset
 
2. [L2: Stochastic block model and mixed-membership](https://github.com/cdebacco/ds_prob_inf_net/blob/main/lectures/L2/L2.pdf)
   - LO2.1: **Infer** hidden mixed-membership clusters or communities from a network dataset
   - LO2.2: **Evaluate** performance in clustering nodes in a network 
   - LO2.3: **Design** a SBM model that incorporates extra parameters or extra data  

3. [L3: Variational inference](https://github.com/cdebacco/ds_prob_inf_net/blob/main/lectures/L3/L3.pdf)
   - LO3.1: **Explain** how variational inference works
   - LO3.2: **Infer** hidden mixtures of Gaussians using variational inference
   - LO3.3: **Derive** CAVI updates for conjugate models
     
4. [L4: Variational inference and mixed-membership models](https://github.com/cdebacco/ds_prob_inf_net/blob/main/lectures/L4/L4.pdf)
   - LO4.1: **Derive** variational inference updates for mixed membership models
   - LO4.2: **Compare** variational inference with MLE + EM in community detection tasks in networks
   - LO4.3: **Analyze** a multilayer network probabilistically 
      
5. [L5: Rankings from pairwise comparisons](https://github.com/cdebacco/ds_prob_inf_net/blob/main/lectures/L5/L5.pdf)
   - LO5.1: **Analyze** a pairwise comparison dataset probabilistically
   - LO5.2: **Infer** hidden scores from pairwise comparisons
   - LO5.3: **Evaluate** different rankings
 
11. [L6: Non-linear models: deep learning architectures for networked datasets](https://github.com/cdebacco/ds_prob_inf_net/blob/main/lectures/L6/L6.pdf)   
