classdef tfw_i < tf_i
  %TFW_I Transformer Wrapper Interface (Base Class)
  %   A transformer that holds internal transform array
  
  properties
    tfs; % for internal transformer array
  end
  
  methods 
    %%% data management
    function ob = cl_io (ob)
      % clear itself
      ob = cl_io@tf_i(ob);
      % clear every transformer
      ob.tfs = cellfun(@cl_io, ob.tfs, 'uniformoutput',false);
    end
    
    function ob = cl_p_d (ob)
      % clear itself
      ob = cl_p_d@tf_i(ob);
      % clear every transformer
      ob.tfs = cellfun(@cl_p_d, ob.tfs, 'uniformoutput',false);     
    end
    
    function ob = to_cpu(ob)
      % convert itself
      ob.ab = arch_cpu();
      ob = cvt_data(ob);
      % convert every transformer
      ob.tfs = cellfun(@to_cpu, ob.tfs, 'uniformoutput',false); 
    end % to_cpu
    
    function ob = to_gpu(ob)
      % convert itself
      ob.ab = arch_gpu();
      ob = cvt_data(ob);
      % convert every transformer
      ob.tfs = cellfun(@to_gpu, ob.tfs, 'uniformoutput',false); 
    end % to_gpu
    %%% auxiliary
    function ob = set_cc(ob, the_cc)
      % set itself
      ob.cc = the_cc;
      % set every transformer
      for i = 1 : numel(ob.tfs)
        ob.tfs{i} = set_cc(ob.tfs{i}, the_cc );
      end % for i
    end % set_cc
    
  end % methods auxiliary
  

end

