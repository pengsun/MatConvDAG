classdef tf_dropout < tf_i
  %TF_DROPOUT Dropout
  %   Detailed explanation goes here
  
  properties
    %is_disabled;
    %is_freezed;
    rate; 
    
    mask;
  end
  
  methods
    function ob = tf_dropout ()
      ob.rate = 0.5;
      
      ob.i = n_data();
      ob.o = n_data();
    end
    
    function ob = fprop(ob)
      if ( ob.cc.is_tr ) % training stage: multiply a random mask
        [ob.o.a, ob.mask] = vl_nndropout(ob.i.a, 'rate',ob.rate);
      else % testing: simply let it pass
        ob.o.a = ob.i.a;
      end
    end % fprop
    
    function ob = bprop(ob)
      if ( ob.cc.is_tr ) % training stage: multiply a random mask
        ob.i.d = vl_nndropout(ob.i.a, ob.o.d, 'mask',ob.mask);
      else % testing
        ob.i.d = ob.o.d; % ?
      end
    end % bprop
    
  end
  
  methods % auxiliary
    function ob = cl_io(ob)
      ob = cl_io@tf_i(ob);
      ob.mask = [];
    end % cl_io
  end
  
end

