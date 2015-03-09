function mnist_small_tr_cpu_lenetTriCon()
%% put all the stuff in a static method if you like
%% init dag: from file or from scratch
beg_epoch = 4;
dir_mo = fullfile(vl_rootnn,'examples_dag2/mo_zoo/mnist_small/cpu_lenetTriCon');
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
h.dir_mo = fullfile(vl_rootnn, 'examples_dag2/mo_zoo/mnist_small/cpu_lenetTriCon');
fn_data  = fullfile(vl_rootnn, 'examples/data/mnist_small_cv5/imdb.mat');
%% do the training
[X, Y] = load_tr_data(fn_data);
train(h, X,Y);

function h = create_dag_from_scratch ()
h = convdag();
h.the_dag = tfw_cpu_lenetTriCon();
  
function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function [X,Y] = load_tr_data(fn_data)
load(fn_data);
ind_tr = find( images.set == 1 );

X = images.data(:,:,:, ind_tr);
Y = images.labels(:, ind_tr);