classdef dag_mb < handle
  %dag_trte A thin wrapper for DAG, training and testing with mini-batch
  %   Detailed explanation goes here
  
  % options
  properties
    beg_epoch; % beggining epoch
    num_epoch; % number of epoches
  end
  
  properties
    the_dag; % the whole DAG as a single transformer
    opt_arr; % numeric optimization array, one for each params(i)
    
    L_tr; % training loss
    cc; % calling context
  end
  
  events
    end_ep;  % end of an epoch
    end_bat; % end of a batch
  end
  
  methods
    function ob = dag_mb()
      ob.beg_epoch = 1; % begining epoch
      ob.num_epoch = 5; % number of epoches
      
      ob.cc = call_cntxt();
    end
    
    function ob = train (ob, tr_bdg)
    % train with the data in tr_bdg
      
      %%% set the ob.the_dag before calling train()
      if ( isempty(ob.the_dag) ), error('set .the_dag first!'); end
      
      ob = prepare_train (ob);
      
      for t = ob.beg_epoch : ob.num_epoch
        % fire: train one epoch
        ob = prepare_train_one_epoch(ob, t);
        ob = train_one_epoch(ob, tr_bdg);
        ob = post_train_one_epoch(ob, t, get_numinst(tr_bdg));
        
        % notify end of eporch
        notify(ob, 'end_ep');
        
      end % for t
      
    end % train
    
    function Ypre = test (ob, te_bdg)
      
      % prepare
      ob = prepare_test(ob);
      
      % initialize a batch generator
      te_bdg = reset_epoch(te_bdg);
      nb = get_numbat(te_bdg);
      
      % test every batch
      % What? Why dividing the testing set into batches? Becuuse this would
      % generate many print infos relieving you while you watch the screen
      for i_bat = 1 : nb
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        % get batch 
        data = get_bd_orig(te_bdg, i_bat);
        
        % set source nodes
        for kk = 1 : numel( ob.the_dag.i )
          ob.the_dag.i(kk).a = data{kk};
        end
        
        % fire: do the batch testing by calling fprop() on each transformer
        ob.the_dag = fprop( ob.the_dag );
        
        % fetch and concatenate the results
        Ypre_bat = reshape(ob.the_dag.get_Ypre(), size(data{2}) );
        if (i_bat==1), Ypre = Ypre_bat;
        else           Ypre = cat(2,Ypre,Ypre_bat); end
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('testing: batch %d of %d, ',...
          i_bat, nb);
        bs = get_bdsz(te_bdg, i_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, bs/t_elapsed);
        
        % notify end of batch
        notify(ob, 'end_bat');
      end % for ii
      
      % notify end of eporch
      notify(ob, 'end_ep');
        
    end % test
  end % methods
  
  methods % auxiliary functions for train
    function ob = prepare_train (ob)

      % the parameters and the corresponding numeric optimizers
      num_params = numel(ob.the_dag.p);
      if ( numel(ob.opt_arr) ~= num_params )
        ob.opt_arr = opt_1storder();
        ob.opt_arr(num_params) = opt_1storder();
      end
      clear num_params;
      
      %%% set calling context
      % for the DAG
      ob.the_dag = set_cc(ob.the_dag, ob.cc);
      % for the optimizers
      for k = 1 : numel(ob.opt_arr)
        ob.opt_arr(k).cc = ob.cc;
      end
      % indicate it's the training stage
      ob.cc.is_tr = true;
      
    end % prepare_train
    
    %%% for training one epoch
    function ob = prepare_train_one_epoch (ob, i_epoch)
      % training
      ob.cc.is_tr = true;
      % set calling context
      ob.cc.epoch_cnt = i_epoch;
      % update the loss
      ob.L_tr(i_epoch) = 0;
    end % prepare_train_one_epoch
    
    function ob = train_one_epoch (ob, tr_bdg)
    % train one epoch
    
      % initialize a batch index generator
      tr_bdg = reset_epoch (tr_bdg);
      nb = get_numbat( tr_bdg );
      
      % train every batch
      for i_bat = 1 : nb
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        % get batch 
        data = get_bd(tr_bdg, i_bat);
        
        % set source nodes
        for kk = 1 : numel( ob.the_dag.i )
          ob.the_dag.i(kk).a = data{kk};
        end
        
        % fire: do the batch training
        ob = prepare_train_one_bat(ob, i_bat, tr_bdg);
        ob = train_one_bat(ob);
        ob = post_train_one_bat(ob, i_bat);
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('epoch %d, batch %d of %d, ',...
          ob.cc.epoch_cnt, ob.cc.iter_cnt, nb);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.cc.batch_sz/t_elapsed);

      end % for ii
      
    end % train_one_eporch
    
    function ob = post_train_one_epoch (ob, i_epoch, varargin)
      % normalize the loss
      N = varargin{1};
      ob.L_tr(end) = ob.L_tr(end) ./ N; 
    end % post_train_one_epoch
    
    %%% for traing one batch
    function ob = prepare_train_one_bat (ob, i_bat, tr_bdg)
      % set calling context
      ob.cc.batch_sz = get_bdsz(tr_bdg, i_bat);
      ob.cc.iter_cnt = i_bat;
    end % prepare_train_one_bat
    
    function ob = train_one_bat (ob)
    % train one batch
    
      %%% fprop & bprop
      ob.the_dag = fprop( ob.the_dag );
      ob.the_dag = bprop( ob.the_dag );
      
      %%% update parameters
      for i = 1 : numel(ob.opt_arr)
        ob.opt_arr(i) = update(ob.opt_arr(i), ob.the_dag.p(i) );
      end
    end % train_one_bat
    
    function ob = post_train_one_bat (ob, i_bat)
      % update the loss
      LL = gather( ob.the_dag.o.a ); % cpu or gpu array
      ob.L_tr(end) = ob.L_tr(end) + sum(LL(:));
    end % post_train_one_bat
    
    %%% for internal data management
    function ob = clear_im_data (ob)
    % clear the intermediate (unnecessary) data: hidden variables .a, .d
    % parameters .d
      
      % clear the input for each transformer
      ob.the_dag = cl_io( ob.the_dag );
      
      % clear .d for all parameters
      % TODO: set a swith here, as sometimes we want save the gradients
      ob.the_dag = cl_p_d( ob.the_dag );
      
    end % clear_im_data
    
  end % methods
  
  methods % auxiliary functions for test
    function ob = prepare_test(ob)
      ob.cc.is_tr = false;
    end % prepare_test
  end % methods    
  
end % convdag

