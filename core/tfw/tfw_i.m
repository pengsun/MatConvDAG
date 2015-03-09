classdef tfw_i < tf_i
  %TFW_I Transformer Wrapper Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    tfs; % for internal transformer array
  end
  
  methods % auxiliary
    
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
    
    function ob = set_cc(ob, varargin)
      if (nargin==1)
        for i = 1 : numel(ob.tfs)
          ob.tfs{i}.cc = ob.cc;
        end % for i
      else
        the_cc = varargin{1};
        % set itself
        ob.cc = the_cc;
        % set every transformer
        for i = 1 : numel(ob.tfs)
          ob.tfs{i} = set_cc(ob.tfs{i}, the_cc );
        end % for i
      end % if
    end % set_cc
    
  end % methods auxiliary
  

end

