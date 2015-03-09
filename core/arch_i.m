classdef arch_i < handle
  %ARCH_I Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods(Abstract)
    rdata = cvt_data(ob, data); 
    % convert data to rdata which is architecture dependent
    
    sync(ob);
    % syncronize
  end
  
end

