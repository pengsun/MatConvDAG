classdef bat_gentor
  %BAT_GENTOR Batch (Index) Generator
  %   Detailed explanation goes here
  
  properties
    num_bat; % number of batches
    i_bat;   % current batch count
  end
  
  properties
    ind_all; % index for all instances (after random permutation)
    bat_sz;  % batch size
  end
  
  methods
    function ob = bat_gentor()
      ob = reset(ob, 888, 168); % Why the numbers?? -_-
    end
    
    function ob = reset(ob, N, bat_sz)
      ob.ind_all = randperm(N);
      ob.bat_sz = min(bat_sz,N);
      
      ob.num_bat = ceil( N / ob.bat_sz );
      ob.i_bat = 1;
    end % reset
    
    function idx = get_idx (ob, varargin)
      if ( isempty(varargin) ) 
        ib = ob.i_bat;
      else
        ib = varargin{1};
      end
      
      i_beg = 1 + (ib - 1) * ob.bat_sz;
      i_end =           ib * ob.bat_sz;
      i_end = min( i_end, numel(ob.ind_all) );
      
      idx = ob.ind_all(i_beg : i_end); 
    end % get_one_bat_ind
    
    function idx = get_idx_orig (ob, varargin)
      if ( isempty(varargin) )
        ib = ob.i_bat;
      else
        ib = varargin{1};
      end
      
      i_beg = 1 + (ib - 1) * ob.bat_sz;
      i_end =           ib * ob.bat_sz;
      i_end = min( i_end, numel(ob.ind_all) );
      
      iall = 1 : numel(ob.ind_all);
      idx = iall(i_beg : i_end);
    end % get_idx_orig
    
  end
  
end

