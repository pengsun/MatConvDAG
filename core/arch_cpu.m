classdef arch_cpu < arch_i
  %ARCH_CPU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    
    function rdata = cvt_data(ob, data)
      rdata = gather(data);
    end
    
    function sync(ob)
    % do nothing  
    end
    
  end
  
end

