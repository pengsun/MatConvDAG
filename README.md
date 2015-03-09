#MatConvDAG
Matlab Convolutional Directed Acyclic (**DAG**) Graph. This project develops on top of [vlfeat/matconvnet](https://github.com/vlfeat/matconvnet) and benefits from the efficient GPU computation APIs therein.

## Purposes/Features
There is already a DAG wrapper in vlfeat/matconvnet, which, however, addresses a slightly different problem domain that this project intends for. Here, the main purposes/features are

- The DAG is represented by a Graph Transformer Network (**GTN**) consisting of interleaved hidden units and transformers. See `./TODO/README.md`
  - The DAG can be built recursively, i.e., a DAG can be seen as a *node* that embeds in a higher level DAG, and so on. This should ease the mannual DAG construction for your own task by simply writing Matlab scripts.
  - The hidden units and paramters are all treated equally. It's up to you on how to initialize them and whether to update them during training. This should ease customized inference, e.g., the image synthesis with a trained model. 
- **Recurrent Network**, which is no more than deep structure with shared 
parameters across layers when unfolding
- Vector-Valued Regression (e.g., for face pose estimation)

## Install
1. Setup the original **MatConvNet** by following the instructions therein. This would compile the mex code, add to path the directory `./matlab`.
2. Setup this project. Simply add directory `./matlab_dag` to path by running in command window the following code:
``` matlab
dag_path.setup;
``` 
or doing this manually (e.g., Matlab Desktop -> File menu -> Set Path)

When it is done, cd to directory `examples_dag` and run the m files for examples. See `examples_dag/README.md` therein.

## TODO
 - DAG/GTN implementations (wrappers)
   - parametric transformer 
     - [x] convolution
   - non-parametric transformer
     - [x] pooling
     - [x] dropout
     - [x] relu
     - [ ] lateral normalization 
   - non-parametric auxiliary transformer
     - [x] multiplex/add
     - [ ] split/concatenate
   - loss transformer 
     - [x] LSE (Least Square Error)
     - [ ] Logit (softmax) 
   - wrapper/example code in `examples_dag`
     - [x] basic training
     - [ ] training with validation
     - [x] A simple DAG other than the pure feed forward net with list structure
   - Miscellaneous
     - [x] GPU version
     - [x] GPU examples
     - [ ] Tighten memory for both CPU and GPU (clear data when fprop and bprop)
 - Extension of `vl_simplenn.m` and associated files
   - [x] Least Square Loss
   - [x] Code for direct CNN Testing

## FIXME
 - [ ] Problematic when batchSize = 1 ?

## Dependecy
This project keeps an eye on [vlfeat/matconvnet](https://github.com/vlfeat/matconvnet) and will update (if needed) whenever there is a new commit. The last vlfeat/matconvnet commit that this project tests on and is compatible with:
TODO

