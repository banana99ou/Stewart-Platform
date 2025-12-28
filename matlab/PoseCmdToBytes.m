classdef PoseCmdToBytes < matlab.System
% PoseCmdToBytes  Convert 6 numeric values into an ASCII command (uint8 bytes).
%
% Use in Simulink via a "MATLAB System" block.
%
% Inputs (6 ports):
%   x, y, z, roll, pitch, yaw (single or double scalars)
%
% Output:
%   y: uint8 column vector containing ASCII bytes:
%      "<x> <y> <z> <roll> <pitch> <yaw>\n"
%
% Notes:
% - Output is FIXED-SIZE (MaxLen x 1) because many Simulink serial send blocks
%   require fixed-size input signals.
% - The unused tail is padded with newline (LF, 10). On the Arduino side, those
%   become empty lines and are ignored by `trim()` + `length()==0` checks.
% - The Stewart sketch reads a line using Serial.readStringUntil('\n') and parses
%   with sscanf("%f %f %f %f %f %f"), so this format is compatible.

  properties (Nontunable)
    % Number of decimal places in the ASCII formatting.
    Precision (1,1) double {mustBeInteger, mustBeNonnegative} = 3

    % Maximum output length (bytes). Must be large enough for worst-case values.
    MaxLen (1,1) double {mustBeInteger, mustBePositive} = 128

    % Byte to pad unused buffer space with. Default is LF (10).
    % Using LF makes the receiver see empty lines, which are typically safe to ignore.
    PadByte (1,1) double {mustBeInteger, mustBeNonnegative, mustBeLessThanOrEqual(PadByte,255)} = 10
  end

  methods (Access = protected)
    function y = stepImpl(obj, x, y, z, roll, pitch, yaw)
      % Six scalar input ports -> one ASCII command line.
      % NOTE (Simulink/codegen): sprintf formatSpec must be a compile-time constant.
      % So we switch on Precision and use literal format strings.
      coder.varsize('s', [1 obj.MaxLen], [false true]); %#ok<EMCA> (codegen-friendly)
      switch obj.Precision
        case 0
          s = sprintf('%.0f %.0f %.0f %.0f %.0f %.0f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
        case 1
          s = sprintf('%.1f %.1f %.1f %.1f %.1f %.1f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
        case 2
          s = sprintf('%.2f %.2f %.2f %.2f %.2f %.2f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
        case 3
          s = sprintf('%.3f %.3f %.3f %.3f %.3f %.3f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
        case 4
          s = sprintf('%.4f %.4f %.4f %.4f %.4f %.4f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
        case 5
          s = sprintf('%.5f %.5f %.5f %.5f %.5f %.5f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
        otherwise
          % Clamp anything >= 6 to 6 dp to avoid excessive length.
          s = sprintf('%.6f %.6f %.6f %.6f %.6f %.6f\n', double(x), double(y), double(z), double(roll), double(pitch), double(yaw));
      end
      b = uint8(s).'; % column

      % Fixed-size output required by many serial send blocks:
      y = repmat(uint8(obj.PadByte), obj.MaxLen, 1);

      if numel(b) > obj.MaxLen
        % Truncate as a safety net (should not happen if MaxLen is set appropriately).
        b = b(1:obj.MaxLen);
        % Ensure there is at least a newline at the end so Arduino can terminate a line.
        b(end) = uint8(10);
      end

      y(1:numel(b)) = b;
    end

    function validateInputsImpl(~, x, y, z, roll, pitch, yaw)
      validateScalarRealNumeric(x, "x");
      validateScalarRealNumeric(y, "y");
      validateScalarRealNumeric(z, "z");
      validateScalarRealNumeric(roll, "roll");
      validateScalarRealNumeric(pitch, "pitch");
      validateScalarRealNumeric(yaw, "yaw");
    end

    function validatePropertiesImpl(obj)
      if obj.Precision > 6
        error('PoseCmdToBytes:Precision', 'Precision must be <= 6 for Simulink/codegen.');
      end
    end

    function sz = getOutputSizeImpl(obj)
      % Fixed-size output.
      sz = [obj.MaxLen 1];
    end

    function dt = getOutputDataTypeImpl(~)
      dt = "uint8";
    end

    function cplx = isOutputComplexImpl(~)
      cplx = false;
    end

    function fix = isOutputFixedSizeImpl(~)
      % Fixed-size (MaxLen x 1).
      fix = true;
    end
  end
end

function validateScalarRealNumeric(v, name)
  if ~(isnumeric(v) && isreal(v) && isscalar(v))
    error('PoseCmdToBytes:InputType', '%s must be a real numeric scalar.', name);
  end
end


