#include <iostream>
#include <cmath>
#include <vector>
#include <ostream>
#include <time.h>

class Neuron {
public:
	double bias;
	std::vector<double> weights;
	std::vector<double> inputs;
	double output = 0.0;

public:
	explicit Neuron (double bias) : bias(bias){ }

	auto calculate_output(const std::vector<double> & inputs) {
		this->inputs = inputs;
		output = squash(calculate_total_net_input());
		return output;
	}

	double calculate_total_net_input() {
		auto total = 0.0;
		for (std::size_t i = 0; i < inputs.size(); ++i) {
			total += inputs[i] * weights[i];
		}
		return total + bias;
	}

			// Apply the logistic function to squash the output of the neuron
			// The result is sometimes referred to as 'net'[2] or 'net'[1]
	double squash(double total_net_input) const {
		return 1.0 / (1.0 + std::exp(-total_net_input));
	}

			// Determine how much the neuron's total input has to change to move closer to the expected output
//
			// Now that we have the partial derivative of the error with respect to the output(∂E / ∂yⱼ) and
			// the derivative of the output with respect to the total net input(dyⱼ / dzⱼ) we can calculate
			// the partial derivative of the error with respect to the total net input.
			// This value is also known as the delta(δ)[1]
			// δ = ∂E / ∂zⱼ = ∂E / ∂yⱼ * dyⱼ / dzⱼ
//
	auto calculate_pd_error_wrt_total_net_input(double target_output) {
		return calculate_pd_error_wrt_output(target_output) * calculate_pd_total_net_input_wrt_input();
	}

	// The error for each neuron is calculated by the Mean Square Error method 
	auto calculate_error(double target_output) {
		return 0.5 * pow(target_output - output, 2);
	}
		// The partial derivate of the error with respect to actual output then is calculated by 
// = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
// = -(target output - actual output)
//
	// The Wikipedia article on backpropagation[1] simplifies to the following, but most other learning material does not [2]
// = actual output - target output
//
		// Alternative, you can use(target - output), but then need to add it during backpropagation[3]
//
		// Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so 
// = ∂E/∂yⱼ = -(tⱼ - yⱼ)
	double calculate_pd_error_wrt_output(double target_output) const {
		return -(target_output - output);
	}
		// The total net input into the neuron is squashed using logistic function to calculate the neuron's output
		// yⱼ = φ = 1 / (1 + e ^ (-zⱼ))
		// Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
//
		// The derivative(not partial derivative since there is only one variable) of the output then is 
	// dyⱼ / dzⱼ = yⱼ * (1 - yⱼ)
	double calculate_pd_total_net_input_wrt_input() const {
		return output * (1.0 - output);
	}

		// The total net input is the weighted sum of all the inputs to the neuron and their respective weights 
// = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
//
	// The partial derivative of the total net input with respective to a given weight(with everything else held constant) then is 
// = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
	auto calculate_pd_total_net_input_wrt_weight(std::size_t index) {
		return inputs[index];
	}

};

class NeuronLayer {
public:
	std::vector<Neuron> neurons;
	double bias;

public:
	explicit NeuronLayer(std::size_t num_neurons, double bias = 0.0) {

		// Every neuron in a layer shares the same bias
		this->bias = bias ? bias : rand()/static_cast<double>(RAND_MAX + 1);;

		neurons.reserve(num_neurons);
		for (std::size_t i = 0; i < num_neurons; ++i) {
			neurons.push_back(Neuron(this->bias));
		}
	}

	auto begin() {return neurons.begin(); }
	auto end() {return neurons.end(); }

	auto feed_forward(const std::vector<double> & inputs) {
		std::vector<double> outputs;
		outputs.reserve(neurons.size());
		for (auto & neuron : neurons) {
			outputs.push_back(neuron.calculate_output(inputs));
		}
		return outputs;
	}
	auto get_outputs() {
		std::vector<double> outputs;
		outputs.reserve(neurons.size());
		for (const auto & neuron : neurons) {
			outputs.push_back(neuron.output);
		}
		return outputs;
	}

};

class NeuralNetwork {
public:
	const double learning_rate = 0.5;

	std::size_t num_inputs = 0;
	NeuronLayer hidden_layer;
	NeuronLayer output_layer;
	using training_set_t = std::vector<std::vector<double>>;

public:
	NeuralNetwork(
		std::size_t num_inputs, 
		std::size_t num_hidden, 
		std::size_t num_outputs, 
		const std::vector<double> & hidden_layer_weights, 
		double hidden_layer_bias,
		const std::vector<double> & output_layer_weights, 
		double output_layer_bias
	) : num_inputs(num_inputs),
		hidden_layer(num_hidden, hidden_layer_bias),
		output_layer(num_outputs, output_layer_bias)
	{
			init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights);
			init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights);
	}

	void init_weights_from_inputs_to_hidden_layer_neurons(const std::vector<double> & hidden_layer_weights) {
		auto weight_num = 0.0;
		for (auto& hidden_neuron : hidden_layer) {
			for (std::size_t i = 0; i < num_inputs; ++i) {
				if (hidden_layer_weights.empty()) {
					hidden_neuron.weights.push_back(rand()/static_cast<double>(RAND_MAX + 1));
				} else {
					hidden_neuron.weights.push_back(hidden_layer_weights[weight_num]);
				}
				weight_num += 1.0;
			}
		}
	}

	void init_weights_from_hidden_layer_neurons_to_output_layer_neurons(const std::vector<double> & output_layer_weights) {
		auto weight_num = 0.0;
		for (auto& output_neuron : output_layer) {
			for (std::size_t i = 0; i < hidden_layer.neurons.size(); ++i) {
				if (output_layer_weights.empty()) {
					output_neuron.weights.push_back(rand()/static_cast<double>(RAND_MAX + 1));
				} else {
					output_neuron.weights.push_back(output_layer_weights[weight_num]);
				}
				weight_num += 1.0;
			}
		}
	}

	auto feed_forward(const std::vector<double> & inputs) {
		auto hidden_layer_outputs = hidden_layer.feed_forward(inputs);
		return output_layer.feed_forward(hidden_layer_outputs);
	}

	// Uses online learning, ie updating the weights after each training case
	void train(const std::vector<double> & training_inputs, const std::vector<double> & training_outputs) {
		feed_forward(training_inputs);

		// 1. Output neuron deltas
		std::vector<double> pd_errors_wrt_output_neuron_total_net_input(output_layer.neurons.size());
		for (std::size_t o = 0; o < output_layer.neurons.size(); ++o) {

			// ∂E / ∂zⱼ
			pd_errors_wrt_output_neuron_total_net_input[o] = output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o]);
		}
		// 2. Hidden neuron deltas
		std::vector<double> pd_errors_wrt_hidden_neuron_total_net_input(hidden_layer.neurons.size());
		for (size_t h = 0; h < hidden_layer.neurons.size(); ++h) {

			// We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
			// dE / dyⱼ = Σ ∂E / ∂zⱼ * ∂z / ∂yⱼ = Σ ∂E / ∂zⱼ * wᵢⱼ
			double d_error_wrt_hidden_neuron_output = 0;
			for (std::size_t o = 0; o < output_layer.neurons.size(); ++o) {
				d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * output_layer.neurons[o].weights[h];
			}
			// ∂E / ∂zⱼ = dE / dyⱼ * ∂zⱼ / ∂
			pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input();
		}
		// 3. Update output neuron weights
		for (std::size_t o = 0; o < output_layer.neurons.size(); ++o) {
			for (std::size_t w_ho = 0; w_ho < output_layer.neurons[o].weights.size(); ++w_ho) {

				// ∂Eⱼ / ∂wᵢⱼ = ∂E / ∂zⱼ * ∂zⱼ / ∂wᵢⱼ
				auto pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho);

				// Δw = α * ∂Eⱼ / ∂wᵢ
				output_layer.neurons[o].weights[w_ho] -= learning_rate * pd_error_wrt_weight;
			}
		}
		// 4. Update hidden neuron weights
		for (size_t h = 0; h < hidden_layer.neurons.size(); ++h) {
			for (size_t w_ih = 0; w_ih < hidden_layer.neurons[h].weights.size(); ++w_ih) {

				// ∂Eⱼ / ∂wᵢ = ∂E / ∂zⱼ * ∂zⱼ / ∂wᵢ
				auto pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih);

				// Δw = α * ∂Eⱼ / ∂wᵢ
				hidden_layer.neurons[h].weights[w_ih] -= learning_rate * pd_error_wrt_weight;
			}
		}
	}


	auto calculate_total_error(const std::vector<training_set_t> & training_sets) {
		auto total_error = 0.0;
		for (std::size_t t = 0; t < training_sets.size(); ++t) {
			auto training_inputs = training_sets[t].front();
			auto training_outputs = training_sets[t].back();

			feed_forward(training_inputs);
			for (std::size_t o = 0; o < training_outputs.size(); ++o) {
				total_error += output_layer.neurons[o].calculate_error(training_outputs[o]);
			}
		}
		return total_error;
	}
};



int main() {
	srand(time(nullptr));
	auto nn = NeuralNetwork(
		2, 2, 2, 
		{0.15, 0.2, 0.25, 0.3}, 
		0.35, 
		{0.4, 0.45, 0.5, 0.55},
		0.6
	);
	std::cout.precision(9);
	for (std::size_t i = 0; i < 10'000; ++i) {
		nn.train({0.05, 0.1}, {0.01, 0.99});
		std::cout << i << " " << nn.calculate_total_error({{{0.05, 0.1}, {0.01, 0.99}}}) << '\n';
	}
}