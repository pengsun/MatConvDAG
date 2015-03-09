classdef tf_cat < tf_i
  %TF_CAT Concatenator
  %   This transformer has only one output and M inputs with sizes
  %   [A1,B1,C1,N],...,[AM,BM,CM,N] respectively. Each input is first
  %   vectorized/reshaped as [1,1,A1*B1*C1,N],...,[1,1,AM*BM*CM,N] and then
  %   concatenated at output with size [1,1,K,N] where K = A1*B1*C1 + ...
  %   + AM*BM*CM.
  %   

  
  properties
  end
  
  methods
    function ob = tf_cat(num_i)
    % Input:
    %  num_i: [1], number of outputs
    
      num_i = getValidArg (ob, num_i);
        
      ob.i        = n_data();
      ob.i(num_i) = n_data();
      ob.o = n_data();
    end
    
    function ob = fprop(ob)
      
      % concatenate each input
      for j = 1 : numel(ob.i)
        aa = ob.i(j).a;
        sz = size(aa); % [Aj,Bj,Cj, N]
        Min = sz(1)*sz(2)*sz(3); N = sz(4);
        
        if (j==1)
          ob.o.a = reshape(aa,[1,1,Min,N]);
        else
          ob.o.a = cat(3, ob.o.a, reshape(aa,[1,1,Min,N]) );
        end
      end % for j
      
    end % fprop
    
    function ob = bprop(ob)

      % distribute to each input and be careful with the size
      ind_base = 0;
      for j = 1 : numel(ob.i)
        sz = size( ob.i(j).a ); % [Aj,Bj,Cj, N]
        M = sz(1)*sz(2)*sz(3);  % N = sz(4);
        
        ind = ind_base + (1:M); % index for input j at the output
        ob.i(j).d = reshape(ob.o.d(1,1,ind,:), sz);
        
        ind_base = ind_base + M;
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

