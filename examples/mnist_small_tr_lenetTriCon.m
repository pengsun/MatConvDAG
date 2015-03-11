function mnist_small_tr_lenetTriCon()
%% init dag: from file or from scratch
beg_epoch = 8;
dir_mo = fullfile(dag_path.root,'examples2/mo_zoo/mnist_small/lenetTriCon');
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
fn_data  = fullfile(dag_path.root, 'examples2/data/mnist_small_cv5/imdb.mat');
%% CPU or GPU
% h.the_dag = to_cpu( h.the_dag );
h.the_dag = to_gpu( h.the_dag );
%% peek and do something (printing, plotting, saving, etc)
hpeek = convdag_peek();
% plot training loss
addlistener(h, 'end_ep', @hpeek.plot_loss);
% save model
hpeek.dir_mo = dir_mo;
addlistener(h, 'end_ep', @hpeek.save_mo);
%% do the training
[X, Y] = load_tr_data(fn_data);
train(h, X,Y);

function h = create_dag_from_scratch ()
h = convdag();
h.the_dag = tfw_lenetTriCon();
  
function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function [X,Y] = load_tr_data(fn_data)
if ( ~exist(fn_data,'file') )
  get_and_save_mnist_small(fn_data);
end
load(fn_data);
ind_tr = find( images.set == 1 );

X = images.data(:,:,:, ind_tr);
Y = images.labels(:, ind_tr);