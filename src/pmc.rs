use rand;
use rand::Rng;
use std::os::raw::c_void;
use std::slice::from_raw_parts;

use nalgebra::DMatrix;

#[no_mangle]
pub unsafe extern fn pmc_create(nb_layers: i32, layers: *mut c_void) -> *mut c_void {
	let mut random = rand::thread_rng();
	let layers = from_raw_parts(layers as *mut i32, nb_layers as usize);
	
	let mut weights: Vec<DMatrix<f64>> = Vec::with_capacity(nb_layers as usize);
	weights.push(DMatrix::zeros(0, 0));
	
	for layer in 1..nb_layers as usize {
		let rows = layers[layer - 1] + 1;
		let columns = layers[layer] + 1;
		let values: Vec<f64> = (0..rows * columns).map(|_| random.gen_range(-0.9, 1.1)).collect::<>();
		
		weights.push(DMatrix::from_row_slice(rows as usize, columns as usize, values.as_ref()));
	}
	
	Box::into_raw(Box::new(weights)) as *mut c_void
}

#[no_mangle]
pub unsafe extern fn pmc_train(
	nb_layers: i32, layers: *mut c_void, model: *mut c_void,
	inputs_size: i32, inputs: *mut c_void,
	outputs_size: i32, outputs: *mut c_void,
	is_regression: i32
) {
	let weights = &mut *(model as *mut Vec<DMatrix<f64>>);
	let layers = from_raw_parts(layers as *mut i32, nb_layers as usize);
	let inputs = from_raw_parts(inputs as *mut f64, inputs_size as usize);
	let outputs = from_raw_parts(outputs as *mut f64, outputs_size as usize);
	
	let prefixed_inputs = prefix_array(inputs);
	let prefixed_outputs = prefix_array(outputs);
	
	let neurons_output = compute_neurons_output(nb_layers, layers, weights, &prefixed_inputs, is_regression);
	let neurons_delta = compute_neurons_delta(nb_layers, layers, weights, &neurons_output, &prefixed_outputs, is_regression);
	update_weights(nb_layers, layers, weights, &neurons_output, &neurons_delta);
}

fn prefix_array(inputs: &[f64]) -> Vec<f64> {
	let mut vector = Vec::new();
	vector.push(1.);
	
	for value in inputs {
		vector.push(value.clone());
	}
	
	vector
}

unsafe fn compute_neurons_output(nb_layers: i32, layers: &[i32], weights: &Vec<DMatrix<f64>>, prefixed_inputs: &Vec<f64>, is_regression: i32) -> Vec<Vec<f64>> {
	let nb_layers = nb_layers as usize;
	
	let mut output: Vec<Vec<f64>> = Vec::new();
	output.push(prefixed_inputs.clone());
	
	for l in 1..nb_layers {
		let nb_neurons = layers[l] as usize + 1;
		let mut layer_output: Vec<f64> = Vec::with_capacity(nb_neurons);
		layer_output.push(1.);
		
		for j in 1..nb_neurons {
			let nb_neurons_previous_layer = layers[l - 1] as usize + 1;
			let mut result: f64 = 0.;
			
			for i in 0..nb_neurons_previous_layer {
				let weight = weights[l].get_unchecked(i, j);
				let neuron_output = output[l - 1][i];
				
				result += weight * neuron_output;
			}
			
			match is_regression == 1 && l == nb_layers - 1 {
				true => layer_output.push(result),
				false => layer_output.push(result.tanh())
			};
		}
		
		output.push(layer_output);
	}
	
	//println!("Neurons output: {:?}", output);
	output
}

unsafe fn compute_neurons_delta(nb_layers: i32, layers: &[i32], weights: &Vec<DMatrix<f64>>, neurons_output: &Vec<Vec<f64>>, expected_output: &Vec<f64>, is_regression: i32) -> Vec<Vec<f64>> {
	let nb_layers = nb_layers as usize;
	let mut deltas: Vec<Vec<f64>> = neurons_output.clone();
	
	// Last layer deltas
	let nb_neurons = layers[nb_layers - 1] + 1;
	for j in 0..nb_neurons as usize {
		let value = neurons_output[nb_layers - 1][j];
		let expected_value = expected_output[j];
		
		deltas[nb_layers - 1][j] =
			if is_regression == 1 { (value - expected_value) }
			else { (1. - value.powf(2.)) * (value - expected_value) }
	}
	
	// Propagation
	for l in 1..nb_layers {
		let layer_index = nb_layers - 1 - l;
		let next_layer_index = layer_index + 1;
		let nb_neurons_next_layer = layers[next_layer_index];
		
		for i in 0..layers[layer_index] as usize + 1 {
			let lhs = 1. - neurons_output[layer_index][i].powf(2.);
			let mut rhs = 0.;
			
			for j in 1..nb_neurons_next_layer as usize + 1 {
				rhs += weights[next_layer_index].get_unchecked(i, j) * deltas[next_layer_index][j];
			}
			
			deltas[layer_index][i] = lhs * rhs;
		}
	}
	
	//println!("Deltas: {:?}", deltas);
	deltas
}

unsafe fn update_weights(nb_layers: i32, layers: &[i32], weights: &mut Vec<DMatrix<f64>>, neurons_output: &Vec<Vec<f64>>, neurons_delta: &Vec<Vec<f64>>) {
	let nb_layers = nb_layers as usize;
	let alpha = 0.1;
	
	//println!("Weights: {:?}", weights);
	
	for l in 1..nb_layers {
		let nb_neurons = layers[l] + 1;
		
		for j in 0..nb_neurons as usize {
			let nb_neurons_previous_layer = layers[l - 1] + 1;
			
			for i in 0..nb_neurons_previous_layer as usize {
				let updated_weight = weights[l].get_unchecked(i, j) - (alpha * neurons_output[l - 1][i] * neurons_delta[l][j]);
				*weights[l].get_unchecked_mut(i, j) = updated_weight;
			}
		}
	}
	
	//println!("Updated weights: {:?}", weights);
}

#[no_mangle]
pub unsafe extern fn pmc_compute(
	nb_layers: i32, layers: *mut c_void, model: *mut c_void,
	inputs_size: i32, inputs: *mut c_void,
	is_regression: i32
) -> *mut c_void {
	let weights = &mut *(model as *mut Vec<DMatrix<f64>>);
	let layers = from_raw_parts(layers as *mut i32, nb_layers as usize);
	
	let inputs = from_raw_parts(inputs as *mut f64, inputs_size as usize);
	let prefixed_inputs = prefix_array(inputs);
	
	let neurons_output = compute_neurons_output(nb_layers, layers, weights, &prefixed_inputs, is_regression);
	let mut output = neurons_output[neurons_output.len() - 1].clone();
	output.remove(0);
	
	//println!("Computed output: {:?}", output);
	Box::into_raw(Box::new(output)) as *mut c_void
}

#[no_mangle]
pub unsafe extern fn pmc_value(values: *mut c_void, index: i32) -> f64 {
	let values = &*(values as *mut Vec<f64>);
	values[index as usize]
}

#[cfg(test)]
mod test {
	use std::os::raw::c_void;
	use nalgebra::DMatrix;
	use super::*;
	
	#[test]
	fn should_create_pmc() {
		let nb_layers = 3;
		let mut layers = [2, 3, 1];
		
		unsafe {
			let layers = layers.as_mut_ptr() as *mut c_void;
			let model = pmc_create(nb_layers, layers);
			let weights = &*(model as *mut Vec<DMatrix<f64>>);
			
			assert_eq!(3, weights.len());
			
			for index in 0..nb_layers as usize {
				let matrix = &weights[index];
				
				if index == 0 {
					assert_eq!(&DMatrix::zeros(0, 0), matrix);
				}
				else {
					for value in matrix.data.iter() {
						assert!(&-0.9 <= value && value <= &1.1)
					}
				}
			}
		}
	}
	
	#[test]
	fn should_train_and_compute_pmc() {
		let nb_layers = 3;
		let mut layers = [2, 3, 1];
		
		let nb_inputs = 2;
		let mut inputs = [1., 2.];
		
		let nb_outputs = 1;
		let mut outputs = [1.];
		
		unsafe {
			let layers = layers.as_mut_ptr() as *mut c_void;
			let inputs = inputs.as_mut_ptr() as *mut c_void;
			let outputs = outputs.as_mut_ptr() as *mut c_void;
			
			let model = pmc_create(nb_layers, layers);
			pmc_train(nb_layers, layers, model, nb_inputs, inputs, nb_outputs, outputs, -1);
			
			let output = pmc_compute(nb_layers, layers, model, nb_inputs, inputs, -1);
			let value = pmc_value(output, 0);
			
			assert!(-1. <= value && value <= 1.);
		}
	}
}