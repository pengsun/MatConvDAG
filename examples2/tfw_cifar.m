classdef tfw_cifar < tfw_i
  %tfw_cifar Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods 
    
    function ob = tfw_cifar()
    % Initialize the DAG net
    
      %%% set the connection structure
      % 1: conv, param
      tfs{1}        = tf_conv();
      % 2: pool
      tfs{2}        = tf_pool();
      tfs{2}.i      = tfs{1}.o;
      % 3: relu
      tfs{3}        = tf_relu();
      tfs{3}.i      = tfs{2}.o;
      % 4: conv, param
      tfs{4}        = tf_conv();
      tfs{4}.i      = tfs{3}.o;
      % 5: relu
      tfs{5}        = tf_relu();
      tfs{5}.i      = tfs{4}.o;
      % 6: pool
      tfs{6}        = tf_pool();
      tfs{6}.i      = tfs{5}.o;
      % 7: conv
      tfs{7}        = tf_conv();
      tfs{7}.i      = tfs{6}.o;
      % 8: relu
      tfs{8}        = tf_relu();
      tfs{8}.i      = tfs{7}.o;
      % 9: pool
      tfs{9}        = tf_pool();
      tfs{9}.i      = tfs{8}.o;
      % 10: ip1
      tfs{10}        = tf_conv();
      tfs{10}.i      = tfs{9}.o;
      % 11: ip2
      tfs{11}        = tf_conv();
      tfs{11}.i      = tfs{10}.o;
      % 12: loss
      tfs{12}      = tf_loss_lse();
      tfs{12}.i(1) = tfs{11}.o;
      % write back
      ob.tfs = tfs;      
      
      %%% input/output data
      ob.i = [n_data(), n_data()]; % X_bat, Y_bat, respectively
      ob.o = n_data();             % the loss
      
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
      
    end % tfw_lenetDropout
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{1}.i.a     = ob.ab.cvt_data( ob.i(1).a ); %
       ob.tfs{12}.i(2).a = ob.ab.cvt_data( ob.i(2).a ); %
       
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

