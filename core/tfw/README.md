#TFW: transformer wrapper

## Design Concept
A transformer can be ssen as a DAG, or vice versa, a DAG can be seen as a
big transformer. In this directory, a couple of predefined transformers whose names start with `tfw_` are provided. It would be more convenient for your own code to directly call those `tfw`s that have been proved to be effective in practice. Examples include:

- convolutional layer + pooling
- linear layer + Relu + Dropout
- TODO

Also, define customized `tfw_xxx` to make your code look more concise!

## How to
When customizing your own `tfw`, just treat it as usual transformer by deriving it from base class `tf_i.m`. Then do the following:
 
- Mannually set the internal connection of the `n_data()` and other `tf_xxx` in the constructor
- Override the `fprop()` and `bprop()`. **Note**: copy the outer and inner `n_data()` properly before calling `fprop()` and `bprop()`. See `tfw_LinReluDrop.m` for an example.

That's it! 
