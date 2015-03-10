classdef tf_conv_mask < tf_conv
  %TF_CONV Convolution with Mask; Can be used for local/sparse connection 
  %   The mask has 0/1 elements and is with the same size of parameters. 
  %   The mask should not be changed across the training stage, i.e., it 
  %   must be known beforehand.
  %
  %   Note: DONT use this as drop-connect which alters the mask every time
  %   fprop() is called. Use it for local/sparse connection that is
  %   specified mannually beforehand.
  % 
  
  properties
    mask;
  end

  methods
    function ob = tf_conv_mask(varargin)
      ob = ob@tf_conv();
      if (nargin==1)
        ob.mask = varargin{1};
      end
    end % tf_conv_mask
    
    function ob = fprop(ob)
      ob.p(1).a = ob.mask .* ob.p(1).a;
      ob = fprop@tf_conv(ob);
    end % fprop
    
    function ob = bprop(ob)
      ob = bprop@tf_conv(ob);
      ob.p(1).d = ob.mask .* ob.p(1).d;
    end % bprop
    
    function ob = cvt_data(ob)
      % convert internal state
      ob.mask = ob.ab.cvt_data( ob.mask );
      % convert other
      ob = cvt_data@tf_i(ob);
    end % cvt_data
    
  end % methods
  
end

