function cifar_tr()
%% init dag: from file or from scratch
beg_epoch = 4;
dir_mo = fullfile(dag_path.root,'examples2/mo_zoo/cifar/cifar');
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
h.num_epoch = 20;
h.batch_sz = 100;
fn_data  = fullfile(dag_path.root, 'examples2/data/cifar/imdb.mat');
%% initialize parameters
h.the_dag = set_and_init_params(h.the_dag);
%% gpu or cpu
h.the_dag = to_gpu(h.the_dag);
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
h.the_dag = tfw_cifar();
  
function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function h = set_and_init_params(h)
% parameter layer I, conv
h.tfs{1}.p(1).a = 1e-4*randn(5,5,3,32, 'single') ; % kernel
h.tfs{1}.p(2).a = zeros(1, 32, 'single');          % bias
h.tfs{1}.stride = 1;
h.tfs{1}.pad    = 2;
% pool
h.tfs{2}.method = 'max';
h.tfs{2}.pool   = [3 3];
h.tfs{2}.stride = 2;
h.tfs{2}.pad    = [0 1 0 1];
% parameter layer II, conv
h.tfs{4}.p(1).a = 0.01*randn(5,5,32,32, 'single');  % kernel
h.tfs{4}.p(2).a = zeros(1,32,'single');             % bias
h.tfs{4}.stride = 1;
h.tfs{4}.pad    = 2;
% pool
h.tfs{6}.method = 'avg';
h.tfs{6}.pool   = [3 3];
h.tfs{6}.stride = 2;
h.tfs{6}.pad    = [0 1 0 1];
% parameter layer III, conv
h.tfs{7}.p(1).a = 0.01*randn(5,5,32,64, 'single');  % kernel
h.tfs{7}.p(2).a = zeros(1,64,'single');             % bias
h.tfs{7}.stride = 1;
h.tfs{7}.pad    = 2;
% pool
h.tfs{9}.method = 'avg';
h.tfs{9}.pool   = [3, 3];
h.tfs{9}.stride = 2;
h.tfs{9}.pad    = [0 1 0 1];
% parameter layer IV, ip1
h.tfs{10}.p(1).a = 0.1*randn(4,4,64,64, 'single');  % kernel
h.tfs{10}.p(2).a = zeros(1,64,'single');            % bias
h.tfs{10}.stride = 1;
h.tfs{10}.pad    = 0;
% parameter layer V, ip2
h.tfs{11}.p(1).a = 0.1*randn(1,1,64,10, 'single');  % kernel
h.tfs{11}.p(2).a = zeros(1,10,'single');            % bias
h.tfs{11}.stride = 1;
h.tfs{11}.pad    = 0;


function [X,Y] = load_tr_data(fn_data)
if ( ~exist(fn_data,'file') )
  tmp_dir = fullfile( fileparts(fn_data) );
  imdb = get_cifar( tmp_dir ); %#ok<NASGU>
  save(fn_data, '-struct', 'imdb');
  clear imdb;
end

load(fn_data);
ind_tr = find( images.set == 1 );

X = images.data(:,:,:, ind_tr);
Y = images.labels(:, ind_tr);