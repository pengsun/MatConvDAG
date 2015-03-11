function get_and_save_mnist_small( fn )
%GET_CIFAR Summary of this function goes here
%   Detailed explanation goes here

% the function websave is unavailable before R2014b
% do it by JAVA
import org.apache.commons.io.FileUtils;
import java.net.URL;
import java.io.File;

urlstr = 'https://github.com/pengsun/matconvnet/blob/master/examples/data/mnist_small_cv5/imdb.mat?raw=true';
url = URL(urlstr);
fn_dest = File(fn);
try
  fprintf('Downloading from %s\n', urlstr);
  FileUtils.copyURLToFile(url, fn_dest);
catch
  error('Downloading from %s fails, do it mannually please.', url);
end


