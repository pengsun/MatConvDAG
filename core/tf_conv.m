classdef tf_conv < tf_i
  %TF_CONV Convolution
  %   Detailed explanation goes here
  
  properties
    pad;
    stride;
  end
  
  methods
    function ob = tf_conv()
      ob.pad = 0;
      ob.stride = 1;
      
      ob.i = n_data();
      ob.o = n_data();
      ob.p = [n_data(), n_data()];
    end % tf_conv
    
    function ob = fprop(ob)
      w = ob.p(1).a;
      b = ob.p(2).a;
      ob.o.a = vl_nnconv(ob.i.a, w,b, 'pad',ob.pad, 'stride',ob.stride);
    end % tf_conv
    
    function ob = bprop(ob)
      w = ob.p(1).a;
      b = ob.p(2).a;
      delta = ob.o.d;
      [ob.i.d, ob.p(1).d, ob.p(2).d] = vl_nnconv(...
        ob.i.a, w, b, delta, 'pad',ob.pad, 'stride',ob.stride);
    end % tf_conv
    
  end
  
end

