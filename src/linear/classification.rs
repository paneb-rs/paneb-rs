use rand;
use rand::Rng;
use std::os::raw::c_void;
use std::slice::from_raw_parts;

#[no_mangle]
pub unsafe extern fn classification_create(size: i32) -> *mut c_void {
	let mut random = rand::thread_rng();
	let weights: Vec<f64> = (0..size).map(|_| random.gen_range(-1., 1.)).collect::<>();
	
	 Box::into_raw(Box::new(weights)) as *mut c_void
}

#[no_mangle]
pub unsafe extern fn classification_weights(weights: *mut c_void, index: i32) -> f64 {
	let weights = &*(weights as *mut Vec<f64>);
	weights[index as usize]
}

#[no_mangle]
pub unsafe extern fn classification_train(weights: *mut c_void, size: i32, inputs: *mut c_void, expected: i32) {
	let weights = &mut *(weights as *mut Vec<f64>);
	let inputs = from_raw_parts(inputs as *mut f64, size as usize);
	
	if inputs.len() + 1 != weights.len() {
		panic!("Input size does not match model");
	}
	
	let sign = classification_sign(weights, inputs);
	
	if expected != sign {
		classification_update_weights(weights, inputs, expected, sign);
	}
}

#[no_mangle]
pub unsafe extern fn classification_compute(weights: *mut c_void, size: i32, inputs: *mut c_void) -> i32 {
	let weights = &mut *(weights as *mut Vec<f64>);
	let inputs = from_raw_parts(inputs as *mut f64, size as usize);
	
	if inputs.len() + 1 != weights.len() {
		panic!("Input size does not match model");
	}
	
	classification_sign(weights, inputs)
}

fn classification_update_weights(weights: &mut Vec<f64>, inputs: &[f64], expected: i32, sign: i32) {
	let alpha = 0.1;
	let diff = (expected - sign) as f64;
	
	weights[0] = weights[0] + alpha * diff * 1.;
	
	for index in 0..inputs.len() {
		weights[index + 1] = weights[index + 1] + alpha * diff * inputs[index]
	}
}

fn classification_sign(weights: &Vec<f64>, inputs: &[f64]) -> i32 {
	match (weights[0] + inputs[0] * weights[1] + inputs[1] * weights[2]) > 0. {
		true => 1,
		false => -1
	}
}

#[cfg(test)]
mod test {
	use super::*;
	
	#[test]
	fn should_create_model() {
		let size = 3;
		
		unsafe {
			let model = classification_create(size);
			let weights = &*(model as *mut Vec<f64>);
			
			for w in weights.iter() {
				assert!(w >= &-1. && w <= &1.);
			}
		}
	}
}