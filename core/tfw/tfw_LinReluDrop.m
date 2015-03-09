classdef tfw_LinReluDrop < tfw_i
  %TFW_LINRELUDROP Linear layer + Relu + Dropout
  %   Detailed explanation goes here
  properties
  end
  
  methods
    function ob = tfw_LinReluDrop(sz)
    % Input:
    %  sz: [W, H, C, M]. 
    
      %%% internal connection
      f = 0.01;
      % 1: full connection, param
      h = tf_conv();
      h.p(1).a = f*randn(sz, 'single'); % kernel
      h.p(2).a = zeros(1, sz(end), 'single'); % bias
      ob.tfs{1} = h;
      
      % 2: relu
      h = tf_relu();
      h.i = ob.tfs{1}.o;
      h.o = n_data();
      ob.tfs{2} = h;
      
      % 3: dropout
      h = tf_dropout();
      h.i = ob.tfs{2}.o;
      ob.tfs{3} = h;
            
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
      
      %%% set calling context
      ob = set_cc(ob);
      
      %%% input/output data
      ob.i = n_data();
      ob.o = n_data();
      
      %%% calling context
      
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

