#MatConvDAG
Matlab Convolutional Directed Acyclic Graph (**DAG**) . This project develops on top of [vlfeat/matconvnet](https://github.com/vlfeat/matconvnet) and benefits from the efficient GPU computation APIs therein.

## Purposes/Features
There is already a DAG wrapper in vlfeat/matconvnet, which, however, addresses a slightly different problem domain with what this project intends for. Here, the main purposes/features are

- The DAG is represented by a Graph Transformer Network (**GTN**) consisting of interleaved hidden units and transformers. See `./core/README.md`
  - The DAG can be built recursively, i.e., a DAG can be seen as a *node* that is embedded in a higher level DAG, and so on. This should ease the mannual DAG construction for your own task by simply writing Matlab scripts.
  - The hidden units and paramters are all treated equally. It's up to you on how to initialize them and whether to update them during training. This should ease customized inference, e.g., the image synthesis with a trained model. 
- **Recurrent Network**, which is no more than deep structure with shared parameters across layers when unfolding
- Vector-Valued Regression (e.g., for face pose estimation)

## Install
1. Setup the original **MatConvNet** by following the instructions therein. This would compile the mex code, add to path the directory `./matlab`.
2. Setup this project. Simply add directory `./core` to path by running in command window the following code:
``` matlab
dag_path.setup;
``` 
or doing this manually (e.g., Matlab Desktop -> File menu -> Set Path)

When it is done, cd to directory `examples` and run the m files for examples. See `examples/README.md` therein.


## Conventions and Workflow
This project always adopts SGD training with mini-batch and hence takes a **Data-Net-Manager** workflow:
* **Data**: use the data batch generator `bdg_xxx` to produce mini-batch fed to the net. Write your own customized data generator by deriving from `bdg_i.m` if necessary (e.g., read image files in a directory or from a remote database).
* **Net**: i.e., the DAG. Create the DAG (including the loss) by manually connecting transformers `tf_xxx()` and hidden units `n_data()`.
* **Manager**: use `dag_mb` to run the training and testing routines. 

Every thing is plug-and-play. 

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
     - [ ] Tight memory for both CPU and GPU (clear data when fprop and bprop)

## FIXME
 - [ ] Problematic when batchSize = 1 ?

## Dependecy
This project keeps an eye on [vlfeat/matconvnet](https://github.com/vlfeat/matconvnet) and will update (if needed) whenever there is a new commit. The last vlfeat/matconvnet commit that this project tests on and is compatible with:

[Commits on Mar 26, 2015](https://github.com/vlfeat/matconvnet/commit/6200ef3aff6a6211dffdc60522be9b9bd9cbb461)

6200ef3


