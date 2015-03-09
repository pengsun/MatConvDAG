classdef tf_i
  %TF_I Transformer Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    p; % parameters
    i; % input variables
    o; % output variables
    
    cc; % calling context
  end
  
  methods
    function ob = tf_i()
      ob.cc = call_cntxt();
    end
    
    function ob = fprop(ob)
    end
    
    function ob = bprop(ob)
    end
    
  end
  
  methods % auxiliary
    function ob = cl_io(ob)
    % clear input, output data
      for k = 1 : numel(ob.i)
        ob.i(k).a = [];
        ob.i(k).d = [];
      end % for k
      for k = 1 : numel(ob.o)
        ob.o(k).a = [];
        ob.o(k).d = [];
      end % for k
    end % cl_io
    
    function ob = cl_i_a(ob)
    % clear .a at each input
      for k = 1 : numel(ob.i)
        ob.i(k).a = [];
      end % for k
    end % cl_i_a
    
    function ob = cl_o_d(ob)
    % clear .d at each output
      for k = 1 : numel(ob.o)
        ob.o(k).d = [];
      end % for k
    end % cl_o_d
    
    function ob = cl_p_d(ob) 
      for k = 1 : numel(ob.p)
        ob.p(k).d = [];
      end % for k
    end % cl_p_d
    
    function ob = set_cc(ob, the_cc)
      ob.cc = the_cc;
    end % set_cc
  end % methods auxiliary
  
end

