classdef tf_loss_lse < tf_i
  %TF_LOSS_LSE Least Square Error for input 1 and input 2
  %   Typically, Input port 1: the prediction, 
  %   Input port 2: the ground truth label
  
  properties
    res; % residual
    sz;  % size for the data at input 1
  end
  
  methods
    function ob = tf_loss_lse()
      ob.i = [n_data(), n_data()];
      ob.o = n_data();
    end % tf_loss_lse
    
    function ob = fprop(ob)
      pre = squeeze(ob.i(1).a);
      tar = ob.i(2).a;
      ob.res = pre - tar;
      
      ob.sz  = size( ob.i(1).a );
      ob.res = reshape(ob.res, ...
        [prod(ob.sz(1:end-1)), ob.sz(end)] ); 
      
      ob.o.a = 0.5 * sum( (ob.res).^2, 1 ); 
    end % fprop
    
    function ob = bprop(ob)
      % just using the "cache"
      %ob.i(1).d = (ob.res) .* ob.o.d;  
      ob.i(1).d = reshape(ob.res, ob.sz );
    end % bprop
    
  end
  
end

