classdef tf_pool < tf_i
  %TF_POOL Pooling
  %   Detailed explanation goes here
  
  properties
    pool;
    pad;
    stride;
    method; 
  end
  
  methods
    function ob = tf_pool()
      ob.pool = [2,2];
      ob.pad = 0;
      ob.stride = 2;
      ob.method = 'max';
      
      ob.i = n_data();
      ob.o = n_data();
    end
    
    function ob = fprop(ob)
      ob.o.a = vl_nnpool(ob.i.a, ob.pool,...
        'pad',ob.pad, 'stride',ob.stride, 'method',ob.method);
    end % fprop
    
    function ob = bprop(ob)
      ob.i.d = vl_nnpool(ob.i.a, ob.pool, ob.o.d,...
        'pad',ob.pad, 'stride',ob.stride, 'method',ob.method);
    end % bprop
  end % methods
  
end

