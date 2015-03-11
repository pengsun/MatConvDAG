function mnist_small_te_cmp()
%% config 1
ep = 1 : 25;
batch_sz = 128;
dir_mo = fullfile(vl_rootnn,'\examples_dag\mo_zoo\mnist_small\lenet');
fn_data = fullfile(vl_rootnn,'\examples\data\mnist_small_cv5\imdb.mat');
fn_mo_tmpl = 'dag_epoch_%d.mat';
%% test 1 
[ep1,err1] = mnist_small_te_all(ep, batch_sz, dir_mo, fn_data, fn_mo_tmpl);
%% config 2
ep = 1 : 25;
batch_sz = 128;
dir_mo = fullfile(vl_rootnn,'\examples_dag\mo_zoo\mnist_small\gpu_lenet');
fn_data = fullfile(vl_rootnn,'\examples\data\mnist_small_cv5\imdb.mat');
fn_mo_tmpl = 'dag_epoch_%d.mat';
%% test 2 
[ep2,err2] = mnist_small_te_all(ep, batch_sz, dir_mo, fn_data, fn_mo_tmpl);
%% plot
figure;
hold on;
plot(ep1,err1, 'ro-', 'linewidth',2);
plot(ep2,err2, 'b*-', 'linewidth',2);
hold off;
set(gca, 'yscale','log');
grid on;
xlabel('epoches'); 
ylabel('testing classification error');