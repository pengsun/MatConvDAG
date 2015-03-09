classdef arch_factory
  %ARCH_FACTORY Summary of this class goes here
  %   Detailed explanation goes here
  
  methods(Static)
    function ha = create(varargin)
      if (nargin==0)
        ha = arch_factory.create_auto(); return;
      end
      
      m = varargin{1};
      switch m
        case 'cpu'
          ha = arch_cpu();
        case 'gpu'
          ha = arch_gpu();
        otherwise
          error('unknown architecture/platform');
      end
    end % create
    
    function ha = create_auto()
      % TODO: create automatically depending on vlfeat/matconvnet
      ha = arch_cpu();
    end % create_auto
    
  end % methods(Static)
  
  methods
    function ob = arch_factory(varargin)
      error('Use the static method create()');
    end
    
  end % methods(Abstract)
  
end

