function mnist_small_tr_gpu_lenetDropout()
%% put all the stuff in a static method if you like
%% init dag: from file or from scratch
beg_epoch = 100;
dir_mo = fullfile(vl_rootnn,'examples_dag2/mo_zoo/mnist_small/gpu_lenetDropout');
fn_mo = fullfile(dir_mo, sprintf('dag_epoch_%d.mat', beg_epoch-1) );
if ( exist(fn_mo, 'file') )
  h = create_dag_from_file (fn_mo);
else
  beg_epoch = 1; 
  h = create_dag_from_scratch ();
end
%% config 
% TODO: add more properties here
h.beg_epoch = beg_epoch;
h.num_epoch = 200;
h.batch_sz = 128;
h.dir_mo = fullfile(vl_rootnn, 'examples_dag2/mo_zoo/mnist_small/gpu_lenetDropout');
fn_data  = fullfile(vl_rootnn, 'examples/data/mnist_small_cv5/imdb.mat');
%% (re-)initialize parameters
% The parameters can be set when h was constructed.
% They can also be (re)set after h was constructed with customized 
% strategies (e.g., Xaiver, Kaiming He...)
h = init_params(h);
%% choose the numeric optimization algorithms
% A default numeric optimization will be set.
% However, customized optimization can also be set here,
% e.g., layer-wise step size, L-BFGS
h = init_opt(h);
%% do the training
[X, Y] = load_tr_data(fn_data);
train(h, X,Y);

function h = create_dag_from_scratch ()
h = convdag();
h.the_dag = tfw_gpu_lenetDropout();
  
function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function h = init_params(h)
f = 0.01;
% parameter layer I, conv
h.the_dag.p(1).a = gpuArray( f*randn(5,5,1,20, 'single') ); % kernel
h.the_dag.p(2).a = gpuArray( zeros(1, 20, 'single') );      % bias
% parameter layer II, conv
h.the_dag.p(3).a = gpuArray( f*randn(5,5,20,50, 'single') ); 
h.the_dag.p(4).a = gpuArray( zeros(1,50,'single')  );        
% parameter layer III, full connection
h.the_dag.p(5).a = gpuArray( f*randn(4,4,50,500, 'single') ); 
h.the_dag.p(6).a = gpuArray( zeros(1,500,'single')  );        
% parameter layer IV, full connection
h.the_dag.p(7).a = gpuArray( f*randn(1,1,500,10, 'single') ); 
h.the_dag.p(8).a = gpuArray( zeros(1,10,'single')  );        

function h = init_opt(h)
num_params = numel(h.the_dag.p);
h.opt_arr = opt_1storder();
h.opt_arr(num_params) = opt_1storder();
% layer wise setp size
rr = [0.01, 0.005, 0.001, 0.001];
for i = 1 : numel(rr)
  h.opt_arr( 2*(i-1) + 1 ).eta = rr(i);
  h.opt_arr( 2*(i-1) + 2 ).eta = rr(i);
end

function [X,Y] = load_tr_data(fn_data)
load(fn_data);
ind_tr = find( images.set == 1 );

X = images.data(:,:,:, ind_tr);
Y = images.labels(:, ind_tr);