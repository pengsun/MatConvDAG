classdef peek_val < handle
  %peek Observer that do printing, validation
  %   Observer Design Pattern
  
  properties
    v_bdg;
    yy;
  end
  
  methods
    function ob_this = peek_val(v_bdg)
      ob_this.v_bdg = v_bdg;
      ob_this.yy = [];
    end
    
    function plot_val_err(ob, h, ~) % (ob_this, ob, evt)
      %%% prepare: clear the data
      h = clear_im_data(h); 
      
      %%% get the testing error for validation set
      % set test
      h.cc.is_tr = false;
      % initialize a batch generator
      ob.v_bdg = reset_epoch(ob.v_bdg);
      nb = get_numbat(ob.v_bdg);
      % test every batch
      for i_bat = 1 : nb
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % get batch 
        data = get_bd(ob.v_bdg, i_bat);
        % set source nodes
        for kk = 1 : numel( h.the_dag.i )
          h.the_dag.i(kk).a = data{kk};
        end
        
        % fire: do the batch testing by calling fprop() on each transformer
        h.the_dag = fprop( h.the_dag );
        
        % fetch and concatenate the results
        Ypre_bat = squeeze( h.the_dag.get_Ypre() );
        if (i_bat==1), Ypre = Ypre_bat;
        else           Ypre = cat(2,Ypre,Ypre_bat); end
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('validating: batch %d of %d, ', i_bat, nb);
        bs = get_bdsz(ob.v_bdg, i_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, bs/t_elapsed);
      end % for ii
      % restore train
      h.cc.is_tr = true;
      
      %%% plot
      figure(41);
      ob.yy(end+1) = get_cls_err(Ypre, ob.v_bdg.Y);
      
      hold on;
      plot( (1 : numel(ob.yy)), ob.yy, 'rx-', 'linewidth',2);
      %set(gca,'yscale','log');
      hold off;
      xlabel('epoch');
      ylabel('validation error');
      grid on;
      drawnow;
    end 
    
  end % methods
  
end % classdef

