% For learning, write m-files init_surname_forward_relog and 
% init_sur-name_backprop_relog, analogous to the posted m-files 
% forward_relu and backprop_relu but modified to use the relog 
% function instead of relu and to take into account the direct 
% projection from layer 2 to layer 4. Please compute the softmax 
% outside your forward_relog function, so your code calls your forward_relog 
% and then applies softmax to its output. And compute the derivative 
% ∂L/∂v{4} outside your backprop_ relog function and use that derivative 
% as the input to backprop_relog. Implement adam using the posted m-file 
% adam.m, and set the adam η = 0.001 and the other adam hyperparameters to 
% their usual values.

%forward relu function:
function net = p_james_forward_relog(net, x)

net.y{1} = x;
for l = 2:net.n_layers - 1
    net.v{l} = net.W{l}*net.y{l - 1} + net.b{l}; %pre-activation column
    y = max(0, sign(net.v{l}).*log(1 + abs(net.v{l})));
    net.y{l} = y;    
end
% l = net.n_layers;
net.out = net.W{4}*[net.y{2}; net.y{3}] + net.b{4};  % neurons in final layer are affine

end