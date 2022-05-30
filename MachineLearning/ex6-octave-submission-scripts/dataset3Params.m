function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

var_results = zeros(64,3);
error_index = 0;


for C_Testing = [0.01 0.03 0.1 0.3 1, 3, 10 30]
    for Sigma_Testing = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        error_index = error_index + 1;
        model = svmTrain(X, y, C_Testing, @(x1, x2) gaussianKernel(x1, x2, Sigma_Testing));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        var_results(error_index,:) = [C_Testing, Sigma_Testing, prediction_error];     
    end
end

% The maximun result as the item sorted.
sorted_results = sortrows(var_results, 3);
% C sorted
C = sorted_results(1,1);
% sigma sorted
sigma = sorted_results(1,2);

% =========================================================================

end
