function mnist_small_tr_lenetDropout()
%% init dag: from file or from scratch
beg_epoch = 3;
dir_mo = fullfile(dag_path.root,'examples2/mo_zoo/mnist_small/lenetDropout');
fn_mo = fullfile(dir_mo, sprintf('dag_epoch_%d.mat', beg_epoch-1) );
if ( exist(fn_mo, 'file') )
  h = create_dag_from_file (fn_mo);
  flag_from_scratch = false;
else
  beg_epoch = 1; 
  h = create_dag_from_scratch ();
  flag_from_scratch = true;
end
%% config
% TODO: add more properties here
h.beg_epoch = beg_epoch;
h.num_epoch = 200;
batch_sz = 128;
fn_data  = fullfile(dag_path.root,...
  'examples2/data/mnist_small_cv5/imdb.mat');
%% (re-)initialize parameters
% The parameters can be set when h was constructed.
% They can also be (re)set after h was constructed with customized 
% strategies (e.g., Xaiver, Kaiming He...)
if (flag_from_scratch)
  h = init_params(h);
end
%% choose the numeric optimization algorithms
% A default numeric optimization will be set.
% However, customized optimization can also be set here,
% e.g., layer-wise step size, L-BFGS
if (flag_from_scratch)
  h = init_opt(h);
end
%% CPU or GPU
% h.the_dag = to_cpu( h.the_dag );
h.the_dag = to_gpu( h.the_dag );
%% peek and do something (printing, plotting, saving, etc)
hpeek = peek();
% plot training loss
addlistener(h, 'end_ep', @hpeek.plot_loss);
% save model
hpeek.dir_mo = dir_mo;
addlistener(h, 'end_ep', @hpeek.save_mo);
%% peek and do validation
v_bdg = load_v_data(fn_data,batch_sz);
hpeek_v = peek_val( v_bdg );
addlistener(h, 'end_ep', @hpeek_v.plot_val_err);
%% do the training
tr_bdg = load_tr_data(fn_data, batch_sz);
train(h, tr_bdg);

function h = create_dag_from_scratch ()
h = dag_mb();
h.the_dag = tfw_lenetDropout();
  
function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function h = init_params(h)
f = 0.01;
% parameter layer I, conv
h.the_dag.p(1).a = f*randn(5,5,1,20, 'single') ; % kernel
h.the_dag.p(2).a = zeros(1, 20, 'single');      % bias
% parameter layer II, conv
h.the_dag.p(3).a = f*randn(5,5,20,50, 'single'); 
h.the_dag.p(4).a = zeros(1,50,'single');        
% parameter layer III, full connection
h.the_dag.p(5).a = f*randn(4,4,50,500, 'single'); 
h.the_dag.p(6).a = zeros(1,500,'single');        
% parameter layer IV, full connection
h.the_dag.p(7).a = f*randn(1,1,500,10, 'single'); 
h.the_dag.p(8).a = zeros(1,10,'single');        

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

function tr_bdg = load_tr_data(fn_data, bs)
if ( ~exist(fn_data,'file') )
  get_and_save_mnist_small(fn_data);
end
load(fn_data);
ind_tr = find( images.set == 1 );

X = images.data(:,:,:, ind_tr);
Y = images.labels(:, ind_tr);
tr_bdg = bdg_memXd4Yd2(X,Y,bs);

function v_bdg = load_v_data(fn_data, bs)
if ( ~exist(fn_data,'file') )
  get_and_save_mnist_small(fn_data);
end
load(fn_data);
ind_v = find( images.set == 3 ); % testing as validation

X = images.data(:,:,:, ind_v);
Y = images.labels(:, ind_v);
v_bdg = bdg_memXd4Yd2(X,Y,bs);
