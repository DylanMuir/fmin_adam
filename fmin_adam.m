function [x, fval, exitflag, output] = fmin_adam(fun, x0, stepSize, beta1, beta2, epsilon, nEpochSize, options)

% fmin_adam - FUNCTION Adam optimiser, with matlab calling format
%
% Usage: [x, fval, exitflag, output] = fmin_adam(fun, x0 <, stepSize, beta1, beta2, epsilon, nEpochSize, options>)
%
% 'fmin_adam' is an implementation of the Adam optimisation algorithm
% (gradient descent with Adaptive learning rates individually on each
% parameter, with momentum) from [1]. Adam is designed to work on
% stochastic gradient descent problems; i.e. when only small batches of
% data are used to estimate the gradient on each iteration.
%
% 'fun' is a function handle [fCost <, vfCdX>] = @(x <, nIter>) defining
% the function to minimise . It must return the cost at the parameter 'x',
% optionally evaluated over a mini-batch of data. If analytical gradients
% are available (recommended), then 'fun' must return the gradients in
% 'vfCdX', evaluated at 'x' (optionally over a mini-batch). If analytical
% gradients are not available, then complex-step finite difference
% estimates will be used.
%
% To use analytical gradients (default), options.GradObj = 'on'. To force
% the use of finite difference gradient estimates, options.GradObj = 'off'.
%
% 'fun' must be deterministic in its calculation of 'fCost' and 'vfCdX',
% even if mini-batches are used. To this end, 'fun' can accept a parameter
% 'nIter' which specifies the current iteration of the optimisation
% algorithm. 'fun' must return estimates over identical problems for a
% given value of 'nIter'.
%
% Steps that do not lead to a reduction in the function to be minimised are
% not taken.
%
% 'x' will be a set of parameters estimated to minimise 'fCost'. 'fval'
% will be the value returned from 'fun' at 'x'.
%
% 'exitflag' will be an integer value indicating why the algorithm
% terminated:
%     0: An output or plot function indicated that the algorithm should
%        terminate.
%     1: The estimated reduction in 'fCost' was less than TolFun.
%    	2: The norm of the current step was less than TolX.
%     3: The number of iterations exceeded MaxIter.
%     4: The number of function evaluations exceeded MaxFunEvals.
%
% 'output' will be a structure containing information about the
% optimisation process:
%     .stepsize      - Norm of current parameter step
%     .gradient      - Vector of current gradients
%     .funccount     - Number of function calls to 'fun' made so far
%     .iteration     - Current iteration of algorithm
%     .fval          - Current value returned by 'fun'
%     .exitflag      - Flag indicating termination reason
%     .improvement   - Current estimated improvement in 'fun'
%
% The optional parameters 'stepSize', 'beta1', 'beta2' and 'epsilon' are
% parameters of the Adam optimisation algorithm (see [1]). Default values
% of {1e-3, 0.9, 0.999, sqrt(eps)} are reasonable for most problems.
%
% The optional argument 'nEpochSize' specifies how many iterations comprise
% an epoch. This is used in the convergence detection code.
%
% The optional argument 'options' is used to control the optimisation
% process (see 'optimset'). Relevant fields:
%     .Display
%     .GradObj
%     .DerivativeCheck
%     .MaxFunEvals
%     .MaxIter
%     .TolFun
%     .TolX
%     .UseParallel
%
%
% References
% [1] Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic
%        Optimization", ICLR 2015.

% Author: Dylan Muir <dylan.muir@unibas.ch>
% Created: 10th February, 2017

%% - Default parameters

DEF_stepSize = 0.001;
DEF_beta1 = 0.9;
DEF_beta2 = 0.999;
DEF_epsilon = sqrt(eps);

% - Default options
if (isequal(fun, 'defaults'))
   x = struct('Display', 'final', ...
      'GradObj', 'on', ...
      'DerivativeCheck', 'off', ...
      'MaxFunEvals', 1e4, ...
      'MaxIter', 1e6, ...
      'TolFun', 1e-6, ...
      'TolX', 1e-5, ...
      'UseParallel', false);
   return;
end


%% - Check arguments and assign defaults

if (nargin < 2)
   help fmin_adam;
   error('*** fmin_adam: Incorrect usage.');
end


if (~exist('stepSize', 'var') || isempty(stepSize))
   stepSize = DEF_stepSize;
end

if (~exist('beta1', 'var') || isempty(beta1))
   beta1 = DEF_beta1;
end

if (~exist('beta2', 'var') || isempty(beta2))
   beta2 = DEF_beta2;
end

if (~exist('epsilon', 'var') || isempty(epsilon))
   epsilon = DEF_epsilon;
end

if (~exist('options', 'var') || isempty(options))
   options = optimset(@fmin_adam);
end


%% - Parse options structure

numberofvariables = numel(x0);

% - Are analytical gradients provided?
if (isequal(options.GradObj, 'on'))
   % - Check supplied cost function
   if (nargout(fun) < 2) && (nargout(fun) ~= -1)
      error('*** fmin_adam: The supplied cost function must return analytical gradients, if options.GradObj = ''on''.');
   end
   
   bUseAnalyticalGradients = true;
   nEvalsPerIter = 2;
else
   bUseAnalyticalGradients = false;
   
   % - Wrap cost function for complex step gradients
   fun = @(x, nIter)FA_FunComplexStepGrad(fun, x, nIter);
   nEvalsPerIter = numberofvariables + 2;
end

% - Should we check analytical gradients?
bCheckAnalyticalGradients = isequal(options.DerivativeCheck, 'on');

% - Get iteration and termination options
MaxIter = FA_eval(options.MaxIter);
options.MaxIter = MaxIter;
options.MaxFunEvals = FA_eval(options.MaxFunEvals);

% - Parallel operation is not yet implements
if (options.UseParallel)
   warning('--- fmin_adam: Warning: ''UseParallel'' is not yet implemented.');
end


%% - Check supplied function

if (nargin(fun) < 2)
   % - Function does not make use of the 'nIter' argument, so make a wrapper
   fun = @(x, nIter)fun(x);
end

% - Check that gradients are identical for a given nIter
if (~bUseAnalyticalGradients)
   [~, vfGrad0] = fun(x0, 1);
   [~, vfGrad1] = fun(x0, 1);
   
   if (max(abs(vfGrad0 - vfGrad1)) > eps(max(max(abs(vfGrad0), abs(vfGrad1)))))
      error('*** fmin_adam: Cost function must return identical stochastic gradients for a given ''nIter'', when analytical gradients are not provided.');
   end
end

% - Check analytical gradients
if (bUseAnalyticalGradients && bCheckAnalyticalGradients)
   FA_CheckGradients(fun, x0);
end

% - Check user function for errors
try
   [fval0, vfCdX0] = fun(x0, 1);
   
catch mErr
   error('*** fmin_adam: Error when evaluating function to minimise.');
end

% - Check that initial point is reasonable
if (isinf(fval0) || isnan(fval0) || any(isinf(vfCdX0) | isnan(vfCdX0)))
   error('*** fmin_adam: Invalid starting point for user function. NaN or Inf returned.');
end


%% - Initialise algorithm

% - Preallocate cost and parameters
xHist = zeros(numberofvariables, MaxIter+1);%MappedTensor(numberofvariables, MaxIter+1);
xHist(:, 1) = x0;
x = x0;
vfCost = zeros(1, MaxIter);

if (~exist('nEpochSize', 'var') || isempty(nEpochSize))
   nEpochSize = numberofvariables;
end

vfCost(1) = fval0;
fLastCost = fval0;
fval = fval0;

% - Initialise moment estimates
m = zeros(numberofvariables, 1);
v = zeros(numberofvariables, 1);

% - Initialise optimization values
optimValues = struct('fval', vfCost(1), ...
   'funccount', nEvalsPerIter, ...
   'gradient', vfCdX0, ...
   'iteration', 0, ...
   'improvement', inf, ...
   'stepsize', 0);

% - Initial display
FA_Display(options, x, optimValues, 'init', nEpochSize);
FA_Display(options, x, optimValues, 'iter', nEpochSize);

% - Initialise plot and output functions
FA_CallOutputFunctions(options, x0, optimValues, 'init');
FA_CallOutputFunctions(options, x0, optimValues, 'iter');


%% - Optimisation loop
while true
   % - Next iteration
   optimValues.iteration = optimValues.iteration + 1;
   nIter = optimValues.iteration;
   
   % - Update biased 1st moment estimate
   m = beta1.*m + (1 - beta1) .* optimValues.gradient(:);
   % - Update biased 2nd raw moment estimate
   v = beta2.*v + (1 - beta2) .* (optimValues.gradient(:).^2);
   
   % - Compute bias-corrected 1st moment estimate
   mHat = m./(1 - beta1^nIter);
   % - Compute bias-corrected 2nd raw moment estimate
   vHat = v./(1 - beta2^nIter);
   
   % - Determine step to take at this iteration
   vfStep = stepSize.*mHat./(sqrt(vHat) + epsilon);
   
   % - Test step for true improvement, reject bad steps
   if (fun(x(:) - vfStep(:), nIter) <= fval)
      x = x(:) - vfStep(:);
      optimValues.stepsize = max(abs(vfStep));
   end
   
   % - Get next cost and gradient
   [fval, optimValues.gradient] = fun(x, nIter+1);
   vfCost(nIter + 1) = fval;
   optimValues.funccount = optimValues.funccount + nEvalsPerIter;
   
   % - Call display, output and plot functions
   bStop = FA_Display(options, x, optimValues, 'iter', nEpochSize);
   bStop = bStop | FA_CallOutputFunctions(options, x, optimValues, 'iter');
   
   % - Store historical x
   xHist(:, nIter + 1) = x;
   
   % - Update covergence variables
   nFirstCost = max(1, nIter + 1-nEpochSize);
   fEstCost = mean(vfCost(nFirstCost:nIter+1));
   fImprEst = abs(fEstCost - fLastCost);
   optimValues.improvement = fImprEst;
   fLastCost = fEstCost;
   optimValues.fval = fEstCost;
   
   %% - Check termination criteria
   if (bStop)
      optimValues.exitflag = 0;
      break;
   end
   
   if (nIter > nEpochSize)
      if (fImprEst < options.TolFun / nEpochSize)
         optimValues.exitflag = 1;
         break;
      end
      
      if (optimValues.stepsize < options.TolX)
         optimValues.exitflag = 2;
         break;
      end
      
      if (nIter >= options.MaxIter-1)
         optimValues.exitflag = 3;
         break;
      end
      
      if (optimValues.funccount > options.MaxFunEvals)
         optimValues.exitflag = 4;
         break;
      end
   end
end

% - Determine best solution
vfCost = vfCost(1:nIter+1);
[~, nBestParams] = nanmin(vfCost);
x = xHist(:, nBestParams);
fval = vfCost(nBestParams);
exitflag = optimValues.exitflag;
output = optimValues;

% - Final display
FA_Display(options, x, optimValues, 'done', nEpochSize);
FA_CallOutputFunctions(options, x, optimValues, 'done');

end

%% Utility functions

% FA_FunComplexStepGrad - FUNCTION Compute complex step finite difference
% gradient estimates for an analytial function
function [fVal, vfCdX] = FA_FunComplexStepGrad(fun, x, nIter)
   % - Step size
   fStep = sqrt(eps);
   
   % - Get nominal value of function
   fVal = fun(x, nIter);
   
   % - Estimate gradients with complex step
   for (nParam = numel(x):-1:1)
      xStep = x;
      xStep(nParam) = xStep(nParam) + fStep * 1i;
      vfCdX(nParam, 1) = imag(fun(xStep, nIter)) ./ fStep; 
   end
end

% FA_CheckGradients - FUNCTION Check that analytical gradients match finite
% difference estimates
function FA_CheckGradients(fun, x0)
   % - Get analytical gradients
   [~, vfCdXAnalytical] = fun(x0, 1);
   
   % - Get complex-step finite-difference gradient estimates
   [~, vfCdXFDE] = FA_FunComplexStepGrad(fun, x0, 1);
   
   disp('--- fmin_adam: Checking analytical gradients...');
   
   % - Compare gradients
   vfGradDiff = abs(vfCdXAnalytical - vfCdXFDE);
   [fMaxDiff, nDiffIndex] = max(vfGradDiff);
   fTolGrad = eps(max(max(abs(vfCdXFDE), abs(vfCdXAnalytical))));
   if (fMaxDiff > fTolGrad)
      fprintf('   Gradient check failed.\n');
      fprintf('   Maximum difference between analytical and finite-step estimate: %.2g\n', fMaxDiff);
      fprintf('   Analytical: %.2g; Finite-step: %.2g\n', vfCdXAnalytical(nDiffIndex), vfCdXFDE(nDiffIndex));
      fprintf('   Tolerance: %.2g\n', fTolGrad);
      fprintf('   Gradient indicies violating tolerance: [');
      fprintf('%d, ', find(vfGradDiff > fTolGrad));
      fprintf(']\n');
      
      error('*** fmin_adam: Gradient check failed.');
   end
   
   disp('   Gradient check passed. Well done!');
end

% FA_Display - FUNCTION Display the current state of the optimisation
% algorithm
function bStop = FA_Display(options, x, optimValues, state, nEpochSize) %#ok<INUSL>
   bStop = false;

   % - Should we display anything?
   if (isequal(options.Display, 'none'))
      return;
   end

   % - Determine what to display
   switch (state)
      case 'init'
         if (isequal(options.Display, 'iter'))
            fprintf('\n\n%10s   %10s   %10s   %10s\n', ...
               'Iteration', 'Func-count', 'f(x)', 'Improvement', 'Step-size');
            fprintf('%10s   %10s   %10s   %10s   %10s\n', ...
               '----------', '----------', '----------', '----------', '----------');
         end

      case 'iter'
         if (isequal(options.Display, 'iter') && (mod(optimValues.iteration, nEpochSize) == 0))
            fprintf('%10d   %10d   %10.2g   %10.2g   %10.2g\n', ...
               optimValues.iteration, optimValues.funccount, ...
               optimValues.fval, optimValues.improvement, optimValues.stepsize);
         end

      case 'done'
         bPrintSummary = isequal(options.Display, 'final') | ...
            isequal(options.Display, 'iter') | ...
            (isequal(options.Display, 'notify') & (optimValues.exitflag ~= 1) & (optimValues.exitflag ~= 2));

         if (bPrintSummary)
            fprintf('\n\n%10s   %10s   %10s   %10s   %10s\n', ...
               'Iteration', 'Func-count', 'f(x)', 'Improvement', 'Step-size');
            fprintf('%10s   %10s   %10s   %10s   %10s\n', ...
               '----------', '----------', '----------', '----------', '----------');
            fprintf('%10d   %10d   %10.2g   %10.2g   %10.2g\n', ...
               optimValues.iteration, optimValues.funccount, ...
               optimValues.fval, optimValues.improvement, optimValues.stepsize);
            fprintf('%10s   %10s   %10s   %10s   %10s\n', ...
               '----------', '----------', '----------', '----------', '----------');

            strExitMessage = FA_GetExitMessage(optimValues, options);
            fprintf('\nFinished optimization.\n   Reason: %s\n\n', strExitMessage);
         end
   end
end

% FA_CallOutputFunctions - FUNCTION Call output and plot functions during
% optimisation
function bStop = FA_CallOutputFunctions(options, x, optimValues, state)
   bStop = false;

   if (~isempty(options.OutputFcn))
      bStop = bStop | options.OutputFcn(x, optimValues, state);
      drawnow;
   end

   if (~isempty(options.PlotFcns))
      if (iscell(options.PlotFcns))
         bStop = bStop | any(cellfun(@(fh)fh(x, optimValues, state), options.PlotFcns));
      else
         bStop = bStop | options.PlotFcns(x, optimValues, state);
      end
      drawnow;
   end
end

% FA_eval - FUNCTION Evaluate a string or return a value
function oResult = FA_eval(oInput)
   if (ischar(oInput))
      oResult = evalin('caller', oInput);
   else
      oResult = oInput;
   end
end

% FA_GetExitMessage - FUNCTION Return the message describing why the
% algorithm terminated
function strMessage = FA_GetExitMessage(optimValues, options)
   switch (optimValues.exitflag)
      case 0
         strMessage = 'Terminated due to output or plot function.';

      case 1
         strMessage = sprintf('Function improvement [%.2g] less than TolFun [%.2g].', optimValues.improvement, options.TolFun);

      case 2
         strMessage = sprintf('Step size [%.2g] less than TolX [%.2g].', optimValues.stepsize, options.TolX);
         
      case 3
         strMessage = sprintf('Number of iterations reached MaxIter [%d].', options.MaxIter);

      case 4
         strMessage = sprintf('Number of function evaluations reached MaxFunEvals [%d].', options.MaxFunEvals);

      otherwise
         strMessage = 'Unknown termination reason.';
   end
end

% --- END of fmin_adam.m ---
