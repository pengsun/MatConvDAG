function rlab = get_vec_labels(lab)
% convert to vector-valued 0/1 labels
K = max(lab(:));
N = numel( lab(:) );
rlab = zeros(K, N, 'single');
for i = 1 : numel(lab)
  ix = lab(i);
  rlab(ix, i) = 1;
end