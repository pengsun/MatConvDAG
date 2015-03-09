classdef dag_path
  %DAG_PATH Path Routines
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
  end
  
  methods(Static)
    function setup()
      try
        vl_setupnn();
      catch
        error('vlfeat/matconvnet not installed properly, quit.');
      end
      
      dag_path.add();
    end
    
    function add ()
      rp = dag_path.root();
      addpath(fullfile(rp, 'core')) ;
      addpath(fullfile(rp, 'core/tfw')) ;
    end
    
    function rp = root ()
      rp = fileparts(fileparts(mfilename('fullpath'))) ;
    end
  end
  
end

