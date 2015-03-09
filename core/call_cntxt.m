classdef call_cntxt < handle
  %CALL_CNTXT Calling Context
  %   Detailed explanation goes here
  
  properties
    iter_cnt; % iteration count
    epoch_cnt;% epoch count
    batch_sz; % batch size
    is_tr;    % training? or testing?
  end
  
  methods
    function ob = call_cntxt()
      ob.iter_cnt = 1;
      ob.epoch_cnt = 1;
      ob.batch_sz = 1;
      ob.is_tr = true;      
    end % call_cntxt
  end
  
end

