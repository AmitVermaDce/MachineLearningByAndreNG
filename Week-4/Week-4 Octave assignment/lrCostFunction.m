function [J, grad] = lrCostFunction(theta, X, y, lambda)
  
  function J = logisticRegressionRegularizedCost(X,y,theta, lambda)
    trainingExamples = length(y);
    hypothesisValue = sigmoid(X*theta);
    
    J = (-1/trainingExamples) * (y' * log(hypothesisValue) + (1-y)' * log(1-hypothesisValue)) + (lambda/(2*trainingExamples)) * (sum(theta.^2) - theta(1)^2);
    
  endfunction
  
  function gradient = logisticRegressionGradientFunc(theta,X,y)
    trainingExamples = length(y);
    hypothesisValue = sigmoid(X*theta);
    gradient = ((1/trainingExamples) * (X' *(hypothesisValue - y)));
  endfunction
  
  function gradient = regularizedlogisticRegressionGradientFunc(X,y,theta)
    trainingExamples = length(y);
    hypothesisValue = sigmoid(X * theta);
    gradient = logisticRegressionGradientFunc(theta,X,y);
    regularizedParameterValue = (lambda / trainingExamples) * theta;
    regularizedParameterValue(1) = 0;
    gradient += regularizedParameterValue;
  endfunction
    
  J = logisticRegressionRegularizedCost(X, y, theta, lambda);
  grad = regularizedlogisticRegressionGradientFunc(X, y, theta);
  
  
end
