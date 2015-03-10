classdef tfw_MLP < tfw_i
  %TFW_LENETDROPOUT Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods 
    
    function ob = tfw_MLP()
    % Initialize the DAG net
    
      %%% set the connection structure: 784 -- 1000 -- 800 -- 10
      f = 1/100;
      % Layer I
      tfs{1} = tfw_LinReluDrop( [28,28,1,1000] ); 
      % Layer II
      tfs{2}   = tfw_LinReluDrop( [1,1,1000,800] );
      tfs{2}.i = tfs{1}.o;
      % Layer III, output
      tfs{3}        = tf_conv();
      tfs{3}.i      = tfs{2}.o;
      tfs{3}.p(1).a = f*randn(1,1,800,10, 'single'); % kernel
      tfs{3}.p(2).a = zeros(1, 10, 'single'); % bias
      % 4: loss
      tfs{4}      = tf_loss_lse();
      tfs{4}.i(1) = tfs{3}.o;
      ob.tfs = tfs;    
      
      %%% input/output data
      ob.i = [n_data(), n_data()]; % X_bat, Y_bat, respectively
      ob.o = n_data();             % the loss
      
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
      
    end % tfw_lenetDropout
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{1}.i.a    = ob.ab.cvt_data( ob.i(1).a ); %
       ob.tfs{4}.i(2).a = ob.ab.cvt_data( ob.i(2).a ); %
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs )
         ob.tfs{i} = fprop(ob.tfs{i});
         ob.ab.sync();
       end
       
       %%% Internal Output --> Outer Output: set the loss
       ob.o.a = ob.tfs{end}.o.a;
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output: unnecessary here
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop(ob.tfs{i});
        ob.ab.sync();
      end
      
      %%% Internal Input --> Outer Input: unnecessary here
    end % bprop
    
    % help
    function Ypre = get_Ypre(ob)
      Ypre = ob.tfs{end-1}.o.a;
    end % get_pre
  end % methods
  
end % tfw_lenetDropout

