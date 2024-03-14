%My Mapnet File. 

function net = p_james_create_mapnet(n_neurons)

map_width = n_neurons(1:end-1);
field_fraction = 0.25;
net.neurons_dense = n_neurons(end);
net.n_neurons = [map_width.*map_width; net.neurons_dense];  % number of neurons in each layer
net.n_layers = numel(net.n_neurons);  % number of layers including input (4)
net.n_map_layers = numel(map_width); %(3)

[net.W, net.W_, net.W__, net.b, net.b_, net.b__, net.v, net.C, ...
  net.y, net.delta, net.dL_dW, net.dL_db] = deal(cell(net.n_layers, 1)); %Initializes array of values/matrices for each layer. 

for l = 2:net.n_layers
    input_num_neurons = net.n_neurons(l - 1);
    current_num_neurons = net.n_neurons(l);
    N = input_num_neurons * (field_fraction^2); %Number of input neurons projecting to a single neuron in this layer

    if l == net.n_layers
        input_num_neurons = net.n_neurons(l - 1) + net.n_neurons(l - 2);
        N = input_num_neurons;
    end
    m = sqrt(2/N); 
    net.W{l} = m*randn(current_num_neurons, input_num_neurons);  
    net.b{l} = zeros(current_num_neurons, 1);  
    
    net.W_{l} = zeros(current_num_neurons, input_num_neurons);
    net.W__{l} = zeros(current_num_neurons, input_num_neurons);

    net.b_{l} = zeros(current_num_neurons, 1);
    net.b__{l} = zeros(current_num_neurons, 1);
end
net.adam_1t = 1;
net.adam_2t = 1;

    % Define input fields
    for l = 2:net.n_map_layers
        net.C{l} = zeros(size(net.W{l}));  % connection matrix
        field_size = ceil(field_fraction*map_width(l - 1));  % width (& height) of input field
        O = ones(field_size, field_size);
        Z = zeros(map_width(l - 1), map_width(l - 1));
        lowest_top = map_width(l - 1) - field_size + 1;  % lowest row of upstream map that can be the top row of an input field
        ratio = (lowest_top - 1)/(map_width(l) - 1);
        for i = 1:map_width(l)  % for each row of this layer's map
            top = min(lowest_top, 1 + round((i - 1)*ratio));  % top row of input field in upstream map
            for j = 1:map_width(l)  % for each column of this layer's map
                left = min(lowest_top, 1 + round((j - 1)*ratio));  % leftmost column of input field in upstream map
                FIELD = Z;
                FIELD(top:(top + field_size - 1), left:(left + field_size - 1)) = O;  % only this square patch of upstream cells projects to this cell
                field = reshape(FIELD, 1, net.n_neurons(l - 1)); %converts to column vector 
                k = (i - 1)*map_width(l) + j;  % number of current cell in this layer's map
                net.C{l}(k, :) = field;
            end
        end
        net.W{l} = net.C{l}.*net.W{l};
%         net.C{l} = sparse(net.C{l});  % Matlab's sparse datatype can sometimes speed up operations
%         net.W{l} = sparse(net.W{l});  % "
    end  % for l

end