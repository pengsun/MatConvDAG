classdef bdg_i
  %BDG_I Batch Data Generator, interface
  %   Detailed explanation goes here
  
  properties
  end
  
  methods(Abstract)
    ob = reset_epoch(ob)
    % reset for a new epoch
    
    data = get_bd (ob, i_bat)
    % get the i_bat-th batch data
    
    data = get_bd_orig (ob, i_bat)
    % get the i_bat-th batch data, original order
    
    N = get_bdsz (ob, i_bat)
    % get the size of the i_bat-th batch data
    
    nb = get_numbat (ob)
    % get number of batchs in an epoch
    
    ni = get_numinst (ob)
    % get number of the total instances
  end
  
end % 