classdef bdg_memXd4Yd2 < bdg_i
  %BDG_MEMXY Batch Data Generator, load instance X and label Y into memory
  %   X: [d1,d2,d3,N], ndims(X) = 4
  %   Y: [K,N], ndims(Y) = 2
  
  properties
    X; % [d1,d2,d3, N], instances. N = total # 
    Y; % [K, N], labels
    
    hb; % handle to a  bat_gentor
  end
  
  methods
    function ob = bdg_memXd4Yd2 (X,Y, bs)
      ob.X = X;
      ob.Y = Y;
      N = size(Y,2);
      ob.hb = bat_gentor();
      ob.hb = reset(ob.hb, N,bs);
    end 
    
    function ob = reset_epoch(ob)
    % reset for a new epoch
      N = size(ob.Y, 2);
      bs = get_bdsz(ob, 1);
      ob.hb = reset(ob.hb, N,bs);
    end
    
    function data = get_bd (ob, i_bat)
    % get the i_bat-th batch data
      idx = get_idx(ob.hb, i_bat);
      data{1} = ob.X(:,:,:,idx);
      data{2} = ob.Y(:,idx);
    end
    
    function data = get_bd_orig (ob, i_bat)
    % get the i_bat-th batch data
      idx = get_idx_orig(ob.hb, i_bat);
      data{1} = ob.X(:,:,:,idx);
      data{2} = ob.Y(:,idx);
    end
    
    function N = get_bdsz (ob, i_bat)
    % get the size of the i_bat-th batch data
      N = numel( get_idx_orig(ob.hb, i_bat) );
    end
    
    function nb = get_numbat (ob)
    % get number of batchs in an epoch
      nb = ob.hb.num_bat;
    end
    
    function ni = get_numinst (ob)
    % get number of the total instances
      ni = size(ob.Y,2);
    end
    
  end % methods
  
end % bdg_memXd4Yd2

