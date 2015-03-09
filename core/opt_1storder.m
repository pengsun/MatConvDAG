classdef opt_1storder < opt_i
  %OPT_1STORDER First Order Numeric Optimization
  %   Gradient Descent with Momentum, Weight Decay
  
  properties
    mo; % momentum rate
    wd; % weight decay rate
    eta; % step size
    
    delta; % the increment to be added
  end
  
  methods
    function ob = opt_1storder()
      ob.mo = 0.9;
      ob.wd = 0.0005;
      ob.eta = 0.001;
      
      ob.delta = 0; % Okay with a scalar
    end
    
    function ob = update(ob, pa)
      ob.delta = ob.mo .* ob.delta ...              % momentum
                 -ob.eta .* ob.wd .* pa.a ...       % weight decay
                 -ob.eta ./ ob.cc.batch_sz .* pa.d; % gradient
      pa.a = pa.a + ob.delta;
    end
  end
  
end

