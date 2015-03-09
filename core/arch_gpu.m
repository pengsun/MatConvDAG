classdef arch_gpu < arch_i
  %ARCH_GPU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    
    function rdata = cvt_data(ob, data)
      rdata = gpuArray(data);
    end
    
    function sync(ob)
      wait(gpuDevice);
    end
    
  end
  
end

