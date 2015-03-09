classdef tfw_cpu_lenetTriCon < tfw_i
  %tfw_gpu_lenetTriCon Lenet, triangular connection for the last hidden layer
  %   The connection is like Fig 2 of "Deep Learning Face Representation from 
  %   Predicting 10,000 Classes" or Fig 1 of "Traffic Signs and Pedestrians
  %   Vision with Multi-Scale Convolutional Networks".
  %   
  %   Warning: No evidence shows such a connection can boost the 
  %   performance on mnist dataset. This wrapper class is just an example
  %   of how to make a non-trivial Directed Acyclic connection.
  
  properties
  end
  
  methods 
    
    function ob = tfw_cpu_lenetTriCon()
    % Initialize the DAG net
    
      %%% set the connection structure
      f = 1/100;
      % 1: conv, param
      tfs{1}        = tf_conv();
      tfs{1}.p(1).a = f*randn(5,5,1,20, 'single'); % kernel
      tfs{1}.p(2).a = zeros(1, 20, 'single'); % bias
      % 2: pool
      tfs{2}   = tf_pool();
      tfs{2}.i = tfs{1}.o;
      % 3: conv, param
      tfs{3}        = tf_conv();
      tfs{3}.i      = tfs{2}.o;
      tfs{3}.p(1).a = f*randn(5,5,20,50, 'single');
      tfs{3}.p(2).a = zeros(1,50,'single');
      % 4: pool
      tfs{4}   = tf_pool();
      tfs{4}.i = tfs{3}.o;
      % 5: dropout
      tfs{5}   = tf_dropout();
      tfs{5}.i = tfs{4}.o;
      % -- Begin: triangular connection for tfs{5,6,7} 
      % 6: multiplexer
      tfs{6}   = tf_mtx(2);
      tfs{6}.i = tfs{5}.o;
      % 7: conv, param
      tfs{7}        = tf_conv();
      tfs{7}.i      = tfs{6}.o(1);
      tfs{7}.p(1).a = f*randn(3,3,50,60, 'single');
      tfs{7}.p(2).a = zeros(1,60,'single');
      % 8: concatenator
      tfs{8}      = tf_cat(2);
      tfs{8}.i(1) = tfs{7}.o;
      tfs{8}.i(2) = tfs{6}.o(2);
      % -- End: triangular connection for tfs{6,7,8} 
      % 9: full connection, param
      tfs{9}        = tf_conv();
      tfs{9}.i      = tfs{8}.o;
      tfs{9}.p(1).a = f*randn(1,1,1040,10, 'single');
      tfs{9}.p(2).a = zeros(1,10,'single');
      % 10: dropout
      tfs{10}   = tf_dropout();
      tfs{10}.i = tfs{9}.o;
      % 11: loss
      tfs{11}      = tf_loss_lse();
      tfs{11}.i(1) = tfs{10}.o;
      % write back
      ob.tfs = tfs;
      
      %%% initialize the input/output data
      ob.i = [n_data(), n_data()]; % X_bat, Y_bat, respectively
      ob.o = n_data();             % the loss
      
      %%% initialize the parameters
      ob.p = dag_util.collect_params( ob.tfs );
      
    end % tfw_gpu_lenetTriCon
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{1}.i.a     = ob.ab.cvt_data( ob.i(1).a ); % X_bat
       ob.tfs{11}.i(2).a = ob.ab.cvt_data( ob.i(2).a ); % Y_bat
       
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
    
    % helper/shortcut
    function Ypre = get_Ypre(ob)
      Ypre = ob.tfs{end-1}.o.a;
    end % get_pre
  end % methods
  
end % tfw_gpu_lenetTriCon

