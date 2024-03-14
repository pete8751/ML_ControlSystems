% Backpropagation Algorithm

%Backprop code:

function net = p_james_backprop_relog(net, delta_out)
%Delta_out should be DL_dV_4
n_m = size(delta_out, 2);
l = net.n_layers;
%==== LAYER 4
net.delta{l} = delta_out;  % Output Layer Loss derivative DL_dV_4.
net.dL_dW{l} = net.delta{l}*[net.y{l - 2}; net.y{l - 1}]'/n_m;
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
l = l - 1;

%==== LAYER 3
post_activation_delta = (net.W{l + 1}(:, 785:1568))'*net.delta{l + 1};
net.delta{l} = post_activation_delta .* (sign(net.y{l}).*(1 ./ (1 + abs(net.v{l}))));
net.dL_dW{l} = net.C{l}.*(net.delta{l}*net.y{l - 1}'/n_m);
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
l = l - 1;

%==== LAYER 2
post_activation_delta = net.W{l+1}'*net.delta{l+1} + (net.W{l + 2}(:, 1:784))'*net.delta{l + 2};
net.delta{l} = post_activation_delta .* (sign(net.y{l}).*(1 ./ (1 + abs(net.v{l}))));
net.dL_dW{l} = net.C{l}.*(net.delta{l}*net.y{l - 1}'/n_m);
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;


net.delta{1} = net.W{2}'*net.delta{2};

end