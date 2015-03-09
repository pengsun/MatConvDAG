function [err_ep, err] =  mnist_small_te_all(varargin)

% config 
% TODO: add more properties here
if ( nargin==0 )
  ep = 1 : 30;
  batch_sz = 128;
  dir_mo = fullfile(dag_path.root,'\examples\mo_zoo\mnist_small\lenetDropout');
  fn_data = fullfile(dag_path.root,'\examples\data\mnist_small_cv5\imdb.mat');
  fn_mo_tmpl = 'dag_epoch_%d.mat';
elseif ( nargin==5 )
  ep = varargin{1};
  batch_sz = varargin{2};
  dir_mo = varargin{3};
  fn_data = varargin{4};
  fn_mo_tmpl = varargin{5};  
else
  error('Invalid arguments.');
end

% load data
[X, Y] = load_te_data(fn_data);

% print
fprintf('data: %s\n', fn_data);

% plot
err_ep = 0;
err = 1;
figure;
hax = axes;
title(dir_mo);
plot_err(hax, err_ep, err);

for i = 1 : numel(ep)
  % init dag: from file 
  fn_mo = sprintf(fn_mo_tmpl, ep(i));
  ffn_mo = fullfile(dir_mo, fn_mo);
  if ( ~exist(ffn_mo,'file') )
    fprintf('%s not found, break and stop.\n', ffn_mo);
    break; 
  end
  load(ffn_mo, 'ob');
  % get ob from here
 
  ob.batch_sz = batch_sz;
  Ypre = test(ob, X);
  Ypre = gather(Ypre);

  % show the error
  err(1+i) = get_cls_err(Ypre, Y);
  err_ep = [err_ep, ep(i)];
  plot_err(hax, err_ep, err)
  
  % print the error
  fprintf('model: %s\n', fn_mo);
  fprintf('classification error = %d\n', err(end) );
end


function [X,Y] = load_te_data(fn_data)
load(fn_data);
ind_te = find( images.set == 3 );

X = images.data(:,:,:, ind_te);
Y = images.labels(:, ind_te);

function err = get_cls_err(Ypre, Y)
[~, label_pre] = max(Ypre,[], 1);
[~, label]     = max(Y,[],    1);
N = numel(label);
err = sum( label_pre ~= label ) / N;

function plot_err(hax, err_ep, err)
plot(err_ep, err, 'ro-', 'linewidth', 2, 'parent', hax);
xlabel('epoches');
ylabel('testing classification error');
set(hax, 'yscale','log');
grid on;
drawnow;