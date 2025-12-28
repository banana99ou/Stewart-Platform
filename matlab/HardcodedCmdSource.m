classdef HardcodedCmdSource < matlab.System
% HardcodedCmdSource  Output a constant ASCII command as uint8 bytes.
%
% Intended for use in a Simulink "MATLAB System" block.
%
% Output (uint8 column vector):
%   "0 0 0 0 0 0\n"
%
% ASCII bytes:
%   '0' = 48, ' ' = 32, '\n' = 10

  properties (Nontunable)
    % Command text to transmit. Must end with newline if the receiver uses readStringUntil('\n').
    Cmd (1,1) string = "0 0 0 0 0 0" + newline
  end

  properties (Access = private)
    CmdBytes uint8
  end

  methods (Access = protected)
    function setupImpl(obj)
      % Convert once to a fixed byte vector.
      obj.CmdBytes = uint8(char(obj.Cmd)).';
    end

    function y = stepImpl(obj)
      % Emit the same bytes every step.
      y = obj.CmdBytes;
    end

    function resetImpl(~)
    end

    function releaseImpl(~)
    end

    function sz = getOutputSizeImpl(obj)
      sz = [strlength(obj.Cmd) 1];
    end

    function dt = getOutputDataTypeImpl(~)
      dt = "uint8";
    end

    function cplx = isOutputComplexImpl(~)
      cplx = false;
    end

    function fix = isOutputFixedSizeImpl(~)
      fix = true;
    end
  end
end


