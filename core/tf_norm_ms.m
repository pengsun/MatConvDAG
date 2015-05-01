classdef tf_norm_ms < tf_i
  %TF_NORM_MS Data value normalization by taking out mean and dividing the
  %standard deviation
  %   Normalize to the range [-1, +1]
  
  properties
    v_mean; % typically [H,W,C]. value mean for each pixel  
    v_std;  % typically [H,W,C]. value std (standard deviation) for each pixel
  end
  
  methods
    function ob = tf_norm_ms()
      ob.i = n_data();
      ob.o = n_data();
    end % tf_norm_ms
    
    function ob = fprop(ob)
      ob.o.a = bsxfun(@minus,   ob.i.a, ob.v_mean);
      ob.o.a = bsxfun(@rdivide, ob.i.a, ob.v_std);
    end 
    
    function ob = bprop(ob)
      ob.i.d = bsxfun(@rdivide, ob.o.d, ob.v_std);
    end 
    
    %%% data management
    function ob = cvt_data(ob)
      % cvt base class
      ob = cvt_data@tf_i (ob);
      % cvt itself
      ob.v_mean = ob.ab.cvt_data( ob.v_mean );
      ob.v_std  = ob.ab.cvt_data( ob.v_std  );
    end
  end % methods
  
end % classdef tf_norm_ms

