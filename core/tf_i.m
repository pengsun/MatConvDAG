classdef tf_i
  %TF_I Transformer Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    p; % parameters
    i; % input variables
    o; % output variables
    
    cc; % calling context
    
    ab; % architecture behaviours (e.g., cpu or gpu), Bridge Design Pattern
  end
  
  methods(Abstract)
    ob = fprop(ob);
    ob = bprop(ob);
  end
  
  methods
    %%%
    function ob = tf_i()
      ob.cc = call_cntxt();
      ob.ab = arch_cpu();
    end
    
    %%% data managment
    function ob = cl_io(ob)
    % clear input, output data
      for k = 1 : numel(ob.i)
        ob.i(k).a = ob.ab.cvt_data( [] );
        ob.i(k).d = ob.ab.cvt_data( [] );
      end % for k
      for k = 1 : numel(ob.o)
        ob.o(k).a = ob.ab.cvt_data( [] );
        ob.o(k).d = ob.ab.cvt_data( [] );
      end % for k
    end % cl_io
    
    function ob = cl_i_a(ob)
    % clear .a at each input
      for k = 1 : numel(ob.i)
        ob.i(k).a = ob.ab.cvt_data( [] );
      end % for k
    end % cl_i_a
    
    function ob = cl_o_d(ob)
    % clear .d at each output
      for k = 1 : numel(ob.o)
        ob.o(k).d = ob.ab.cvt_data( [] );
      end % for k
    end % cl_o_d
    
    function ob = cl_p_d(ob) 
      for k = 1 : numel(ob.p)
        ob.p(k).d = ob.ab.cvt_data( [] );
      end % for k
    end % cl_p_d
    
    %%% data conversion
    function ob = to_cpu(ob)
      ob.ab = arch_cpu();
      ob = cvt_data(ob);
    end % to_cpu
    
    function ob = to_gpu(ob)
      ob.ab = arch_gpu();
      ob = cvt_data(ob);
    end % to_gpu
    
    function ob = cvt_data(ob)
    % convert input, output and paramters to current platform data  
      for k = 1 : numel(ob.i)
        ob.i(k).a = ob.ab.cvt_data( ob.i(k).a );
        ob.i(k).d = ob.ab.cvt_data( ob.i(k).d );
      end
      for k = 1 : numel(ob.o)
        ob.o(k).a = ob.ab.cvt_data( ob.o(k).a );
        ob.o(k).d = ob.ab.cvt_data( ob.o(k).d );
      end
      for k = 1 : numel(ob.p)
        ob.p(k).a = ob.ab.cvt_data( ob.p(k).a );
        ob.p(k).d = ob.ab.cvt_data( ob.p(k).d );
      end
    end % cvt_data
    
    %%% auxiliary
    function ob = set_cc(ob, the_cc)
      ob.cc = the_cc;
    end % set_cc
  end % methods auxiliary
  
end

