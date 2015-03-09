classdef dag_util
%DAG_UTIL Encapsulate Utility Functions
%   Detailed explanation goes here
methods(Static)
  
  function params = collect_params(tfs)
    params = [];
    for i = 1 : numel(tfs)
      % empty
      if ( isempty(tfs{i}.p) ), continue; end
      
      % tf: fecth them
      if ( isempty(params) )
        params = tfs{i}.p;
      else
        params = [params, tfs{i}.p];
      end
    end % for i
  end % collect_params
  
  
  
end % methods(Static)
end % classdef dag_util

