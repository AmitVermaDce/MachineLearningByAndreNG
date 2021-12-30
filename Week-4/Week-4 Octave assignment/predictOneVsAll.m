function p = predictOneVsAll(all_theta, X)
  trainingDataSize = size(X)(1);

  % Add ones to the X data matrix
  X = [ones(trainingDataSize, 1) X];

  hypotheses = X * all_theta';
  probabilities = sigmoid(hypotheses);
  [maxProbability index] = max(probabilities, [], 2);
  p = index;


% =========================================================================

end
