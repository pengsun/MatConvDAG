# Wrapper for DAG/GTN

Directed Acyclic Graph (**DAG**) convolutional net is a kind of Graph Transformer Network (**GTN**) [1]. Examples include check reader [1], scene parser [??], pose estimation [??], multi resolution CNN by triangular connection [2, 3], Cascade Neural Network [4], etc. The feed-forward net with list data structure is a special case of DAG/GTN. The DAG/GTN was also studied in previous literature under the name of Energy Based Learning.

## Design Concept: Data Structure
The neural network is seen as a _Directed Acyclic Graph_. 

This way, the _node_ represents the _data block_ including hidden variables, instances and labels, parameters, loss, etc. Typically, the instances, labels and parameters should be source/root node without any incoming edge, while the loss should be sink/leaf node without any outgoing edge. 

Also, the _edge_ represents a _layer_ or a _transformer_, including convolutional layer, pooling layer, loss layer, etc. Other auxiliary transformers manipulating data (e.g., multiplexer which at the output replicates each of the inputs) should be used in together with the off-the-shelf ones for your own customized task.  

In this project, the _data block_ is represented by class `n_data.m`, the _transformer_ is represented by `tf_xxx` or `tfw_xx`, where the latter is simply a wrapper of other transformers (See `matlab_dag/tfw/README.md`).  

An `n_data()` always has `.a` and `.d` properties, being the _activation values_ and the _delta signals_ in _feed forward_ and _back propagation_ procedures, respectively. A `tf(w)_xxx` always has `.i` and `.o` properties, being the input and output `n_data()`, respectively. Note that the `n_data()` is a Matlab handle class, permitting the behavior of "passing by reference" or a pointer in C/C++.

The input and output of a `tf(w)_xxx` can be one `n_data()` (typical in a pure feed forward net with list structure) or multiple `n_data()` (typical in a real DAG). However, the `n_data()` must connect to and be connected with **only one** `n_data()`, which significantly simplifies the design for this project. Just use `n_data()` manipulator (e.g., `tf_mtx`, `tf_cat`) when the _node_ in _DAG_ indeed has multiple incoming and outgoing _edges_. Note that the topology of _DAG_ in your paper should not be confused with the topology of `n_data()`-`tf(w)_xxx` connection in your code. :)    

The _parameters_ that need be learned in training are also represented by `n_data()`. The parameters are always updated by SGD (Stochastic Gradient Descent). A standalone class `opt_xxx` would accommodate the specific numeric optimization algorithm (e.g., `opt_1storder.m` the first order method with momentum and weight decay, `TODO` the L-BFGS). 

The wrapping is in an Object Oriented way, which should be hopefully more flexible and much easier to prototype your own idea by combining the the (standard or customized) data and transformers. See `tfw/READEME.md` for examples.

Finally, the Abstraction Penalty (overhead for the wrapping data structure) should be negligible.

## Design Concept: GPU support
It is tempted to add to the DAG wrapper a variable indicating whether to take CPU or GPU computation. This choice might be necessary when writing low-level functions, however, it should be avoided in high-level wrapper as it might introduce many if-else for CPU or GPU and quickly make your code messy. 

In this project we suggest to prepare a separate wrappers for GPU and CPU implementation, respectively. This way, the script code would be clearer. See the classes and scripts in `examples_dag` (slightly deprecated) and `examples_dag2` (more preferred).

## Reference
[1]. LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

[2]. Sermanet, Pierre, Koray Kavukcuoglu, and Yann LeCun. "Traffic signs and pedestrians vision with multi-scale convolutional networks." Snowbird Machine Learning Workshop. Vol. 2. No. 3. 2011.

[3]. Sun, Yi, Xiaogang Wang, and Xiaoou Tang. "Deep learning face representation from predicting 10,000 classes." Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.

[4]. Matlab Neural Network Toolbox.
