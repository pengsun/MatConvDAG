classdef tfw_LinReluDrop < tfw_i
  %TFW_LINRELUDROP Linear layer + Relu + Dropout
  %   Detailed explanation goes here
  properties
  end
  
  methods
    function ob = tfw_LinReluDrop(varargin)
    % Input:
    %  sz: [W, H, C, M]. 
      sz = [0 0 0 0];
      if (nargin==1)
        sz = varargin{1};
      end
      
      %%% internal connection
      f = 0.01;
      % 1: full connection, param
      ob.tfs{1}        = tf_conv();
      ob.tfs{1}.p(1).a = f*randn(sz, 'single');       % kernel
      ob.tfs{1}.p(2).a = zeros(1, sz(end), 'single'); % bias
      
      % 2: relu
      ob.tfs{2}   = tf_relu();
      ob.tfs{2}.i = ob.tfs{1}.o;
      
      % 3: dropout
      ob.tfs{3}   = tf_dropout();
      ob.tfs{3}.i = ob.tfs{2}.o;
            
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
      
      %%% input/output data
      ob.i = n_data();
      ob.o = n_data();
      
    end % tfw_LinReluDrop
    
    function ob = fprop(ob)
      ob.tfs{1}.i.a = ob.i.a; % outer -> inner
      ob.tfs = cellfun(@fprop, ob.tfs, 'uniformoutput',false);
      ob.o.a = ob.tfs{end}.o.a; % inner -> outer
    end % fprop
    
    function ob = bprop(ob)
      ob.tfs{end}.o.d = ob.o.d; % outer -> inner
      ob.tfs(end:-1:1) = cellfun(@bprop, ob.tfs(end:-1:1),...
        'uniformoutput',false);
      ob.i.d = ob.tfs{1}.i.d; % inner -> outer
    end % bprop
    
  end % methods
  
end

