classdef tf_add < tf_i
  %TF_ADD Adder
  %   This transformer has only one output and M inputs with the same size. 
  %   The output is the sum of all inputs. 
  %
  
  properties
  end
  
  methods
    function ob = tf_add(num_i)
    % Input:
    %  num_i: [1], number of outputs
    
      num_i = getValidArg (ob, num_i);
        
      ob.i        = n_data();
      ob.i(num_i) = n_data();
      ob.o = n_data();
    end
    
    function ob = fprop(ob)
 
      ob.o.a = ob.i(1).a;
      for k = 2 : numel(ob.i)
        ob.o.a = ob.o.a + ob.i(k).a;
      end % for j
      
    end % fprop
    
    function ob = bprop(ob)

      % replicate the .d at each input
      for k = 1 : numel(ob.i)
        ob.i(k).d = ob.o.d;
      end % for j
    end % bprop
    
  end % methods
  
  methods % auxiliary functions
    function n = getValidArg(ob, n)
      n = floor(n);
      if (n < 1), n = 1; end
    end % getValidArg
  end % methods auxiliary functions
  
end

