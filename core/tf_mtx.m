classdef tf_mtx < tf_i
  %TF_MTX The Multiplexer
  %   This transformer has only one input and M outputs. It simply
  %   replicates the input at each of its output, hence the name
  %   multiplexer
  
  properties
  end
  
  methods
    function ob = tf_mtx(num_o)
    % Input:
    %  num_o: [1], number of outputs
    
      num_o = getValidArg (ob, num_o);
        
      ob.i = n_data();
      ob.o        = n_data();
      ob.o(num_o) = n_data();
    end
    
    function ob = fprop(ob)
      % out_1 = in,... out_M = in
      for i = 1 : numel(ob.o)
        ob.o(i).a = ob.i.a;
      end % for i
      
    end % fprop
    
    function ob = bprop(ob)
      % in = out_1 + ... + out_M
      ob.i.d = ob.o(1).d;
      for j = 2 : numel(ob.o)
        ob.i.d = ob.i.d + ob.o(j).d;
      end % for i
      
    end % bprop
    
  end % methods
  
  methods % auxiliary functions
    function n = getValidArg(ob, n)
      n = floor(n);
      if (n < 1), n = 1; end
    end % getValidArg
  end % methods auxiliary functions
  
end