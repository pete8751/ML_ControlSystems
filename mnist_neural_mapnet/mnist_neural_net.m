clear variables; 
clc;

%Load Data:
load 'MNIST.mat';

errors = 10000;

for Run = 1:1:30
% Set up task
Epochs = 10;
Training_Num = 60000;
Batch_Size = 100;
n_x = 784;  % #elements in input vector x, a.k.a. y{1}
n_out = 10; % #elements in output vector x, a.k.a. y{1}
net = p_james_create_mapnet([28; 28; 28; n_out]);
n_m = 1;  % examples per minibatch

% Adam Parameters
eta = 0.001;
adam_1 = 0.9;
adam_2 = 0.999;
%Adam_1t, Adam_2t are defined in mapnet initialization.


% disp(['Learning has started, learning rate: ', num2str(eta)]);
for Epoch = 1:1:Epochs
    %Shuffle Data. 
    shuffle = randperm(size(TRAIN_images, 1));
    TRAIN_images = TRAIN_images(shuffle, :);
    TRAIN_labels = TRAIN_labels(shuffle, :);

    for Batch = 1:1:(Training_Num/Batch_Size)
        start = ((Batch - 1) * Batch_Size) + 1;
        finish = Batch * Batch_Size;
        x = TRAIN_images(start: finish, :)';  % Batch'd training matrix
        t = TRAIN_labels(start: finish, :)';  % Batch'd label matrix
        net = p_james_forward_relog(net, x);
        
        out = net.out;
        exponentials = exp(out); %Applying softmax to output.
        softmax_denoms = sum(exponentials, 1);
        softmax_out = exponentials ./ softmax_denoms;

        e = softmax_out - t; % Computing error.
%         total_L = sum(e.*e, 1) / 2; %Computing Average Loss
%         avg_L = mean(total_L, 2);
        %softmax_out *.e results in each column being y_i .* (y_i - t_i). 
        right_terms = sum(softmax_out.*e, 1);
        dL_dv = softmax_out.*e - (softmax_out.*right_terms); 
        net = p_james_backprop_relog(net, dL_dv);

        %ADAM:
        dL_dW = net.dL_dW;
        dL_db = net.dL_db;

        net.adam_1t = net.adam_1t*adam_1;
        net.adam_2t = net.adam_2t*adam_2;
        correction_1 = 1/(1 - net.adam_1t);
        correction_2 = 1/(1 - net.adam_2t);
        for l = 2:net.n_layers
            net.W_{l} = adam_1*net.W_{l} + (1 - adam_1)*dL_dW{l};  % first moment
            net.W__{l} = adam_2*net.W__{l} + (1 - adam_2)*(dL_dW{l}.*dL_dW{l});  % second moment
            denom = sqrt(net.W__{l}*correction_2) + 1e-8;  % *****
            net.W{l} = net.W{l} - ((eta*correction_1) * net.W_{l} ./ denom);  % *****
            net.b_{l} = adam_1*net.b_{l} + (1 - adam_1)*dL_db{l};  % first moment
            net.b__{l} = adam_2*net.b__{l} + (1 - adam_2)*(dL_db{l}.*dL_db{l});  % second moment
            denom = sqrt(net.b__{l}*correction_2) + 1e-8;  % *****
            net.b{l} = net.b{l} - ((eta*correction_1) * net.b_{l} ./ denom);  % *****
        end

    end

    x = TEST_images';  % Test Images Matrix
    t = TEST_answers;    % Test Answers Column
    net = p_james_forward_relog(net, x);

    out = net.out;
    exponentials = exp(out); %Applying softmax to output.
    softmax_denoms = sum(exponentials, 1);
    softmax_out = exponentials ./ softmax_denoms; %output matrix

    [~, maxIndices] = max(softmax_out);
    predictions = (maxIndices - 1)';

    accuracy_vector = predictions - TEST_answers; %subtract prediction column from answers
    Incorrect_predictions = nnz(accuracy_vector); %count number nonzero entries
    if errors > Incorrect_predictions
        errors = Incorrect_predictions;
    end
    disp(['Total Incorrect predictions in Epoch ', num2str(Epoch), ' is ', num2str(Incorrect_predictions)]);
    
end
    disp(['Lowest Incorrect predictions is ', num2str(errors)]);
end

disp('Learning has ended');
