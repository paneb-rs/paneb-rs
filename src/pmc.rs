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
	model: *mut c_void, nb_layers: i32, layers: *mut c_void,
	inputs_size: i32, inputs: *mut c_void,
	outputs_size: i32, outputs: *mut c_void
) {
	let weights = &mut *(model as *mut Vec<DMatrix<f64>>);
	let layers = from_raw_parts(layers as *mut i32, nb_layers as usize);
	let prefixed_inputs = prefix_inputs(from_raw_parts(inputs as *mut f64, inputs_size as usize));
	let _outputs = from_raw_parts(outputs as *mut f64, outputs_size as usize);
	
	let _neurons_output = compute_neurons_output(weights, nb_layers, layers, prefixed_inputs);
	
	// Update output neurones
	// Retropropagation
		// Compute delta last layer
		// Compute delta other layers (backwards)
		// Re-compute weights (forward)
}

fn prefix_inputs(inputs: &[f64]) -> Vec<f64> {
	let mut vector = Vec::new();
	vector.push(1.);
	
	for value in inputs {
		vector.push(value.clone());
	}
	
	println!("Input values: {:?}, Input vector: {:?}", inputs, vector);
	vector
}

unsafe fn compute_neurons_output(weights: &Vec<DMatrix<f64>>, nb_layers: i32, layers: &[i32], prefixed_inputs: Vec<f64>) -> Vec<Vec<f64>> {
	let mut output: Vec<Vec<f64>> = Vec::new();
	output.push(prefixed_inputs);
	
	for l in 1..nb_layers as usize {
		let nb_neurons = layers[l] as usize + 1;
		let mut layer_output: Vec<f64> = Vec::with_capacity(nb_neurons);
		layer_output.push(1.);
		
		println!("Layer n°{}: {} neurons", l, nb_neurons);
		
		for j in 1..nb_neurons as usize {
			let nb_neurons_previous_layer = layers[l - 1] as usize + 1;
			let mut result: f64 = 0.;
			
			println!("Previous layer n°{}: {} neurons", l - 1, nb_neurons_previous_layer);
			
			for i in 0..nb_neurons_previous_layer {
				println!("Previous neuron index {}", i);
				let weight = weights[l].get_unchecked(i, j);
				println!("Weight: {}", weight);
				let neuron_output = output[l - 1][i];
				println!("Neuron output: {}", neuron_output);
				
				result += weight * neuron_output;
			}
			
			layer_output.push(result.tanh());
		}
		
		output.push(layer_output);
	}
	
	println!("Neurons output: {:?}", output);
	output
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
						//print!("{} ", value);
						assert!(&-0.9 <= value && value <= &1.1)
					}
					//println!("");
				}
			}
		}
	}
	
	#[test]
	fn should_train_pmc() {
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
			let untrained_weights = &*(model as *mut Vec<DMatrix<f64>>).clone();
			
			pmc_train(model, nb_layers, layers, nb_inputs, inputs, nb_outputs, outputs);
			let trained_weights = &*(model as *mut Vec<DMatrix<f64>>).clone();
			
			assert_eq!(trained_weights, untrained_weights);
		}
	}
}