classdef tfw_cpu_lenetDropout < tfw_i
  %TFW_LENETDROPOUT Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods 
    
    function ob = tfw_cpu_lenetDropout()
    % Initialize the DAG net
    
      %%% set the connection structure
      f = 1; % intentionally inappropriate ratio
      % 1: conv, param
      tfs{1}        = tf_conv();
      tfs{1}.p(1).a = f*randn(5,5,1,20, 'single'); % kernel
      tfs{1}.p(2).a = zeros(1, 20, 'single'); % bias
      % 2: pool
      tfs{2}   = tf_pool();
      tfs{2}.i = tfs{1}.o;
      tfs{2}.o = n_data();
      % 3: conv, param
      tfs{3}        = tf_conv();
      tfs{3}.i      = tfs{2}.o;
      tfs{3}.p(1).a = f*randn(5,5,20,50, 'single');
      tfs{3}.p(2).a = zeros(1,50,'single');
      % 4: pool
      tfs{4}   = tf_pool();
      tfs{4}.i = tfs{3}.o;
      % 5: full connection, param
      tfs{5}        = tf_conv();
      tfs{5}.i      = tfs{4}.o;
      tfs{5}.p(1).a = f*randn(4,4,50,500, 'single');
      tfs{5}.p(2).a = zeros(1,500,'single');
      % 6: relu
      tfs{6}   = tf_relu();
      tfs{6}.i = tfs{5}.o;
      tfs{6}.o = n_data();
      % 7: dropout
      tfs{7}   = tf_dropout();
      tfs{7}.i = tfs{6}.o;
      tfs{7}.o = n_data();
      % 8: full connection, param
      tfs{8}        = tf_conv();
      tfs{8}.i      = tfs{7}.o;
      tfs{8}.p(1).a = f*randn(1,1,500,10, 'single');
      tfs{8}.p(2).a = zeros(1,10,'single');
      % 9: loss
      tfs{9}      = tf_loss_lse();
      tfs{9}.i(1) = tfs{8}.o;
      %tfs{9}.o.d  = single(1); % Init the sink node, OK with scarler
      ob.tfs = tfs;      
      
      %%% input/output data
      ob.i = [n_data(), n_data()]; % X_bat, Y_bat, respectively
      ob.o = n_data();             % the loss
      
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
      
    end % tfw_lenetDropout
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{1}.i.a    = ob.i(1).a; %
       ob.tfs{9}.i(2).a = ob.i(2).a; %
       
       %%% fprop for all
       ob.tfs = cellfun(@fprop, ob.tfs, 'uniformoutput',false);
       
       %%% Internal Output --> Outer Output: set the loss
       ob.o.a = ob.tfs{end}.o.a;
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output: unnecessary here
      
      %%% bprop for all
      ob.tfs(end:-1:1) = cellfun(@bprop, ob.tfs(end:-1:1),...
        'uniformoutput', false);
      
      %%% Internal Input --> Outer Input: unnecessary here
    end % bprop
    
    % help
    function Ypre = get_Ypre(ob)
      Ypre = ob.tfs{end-1}.o.a;
    end % get_pre
  end % methods
  
end % tfw_lenetDropout

