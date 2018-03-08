use rand;
use rand::Rng;
use std::os::raw::c_void;

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
pub unsafe extern fn classification_train(weights: *mut c_void, x: f64, y: f64, expected: i32) {
	let weights = &mut *(weights as *mut Vec<f64>);
	let actual = classification_sign(weights, x, y);
	
	if expected != actual {
		classification_update_weights(weights, x, y, expected, actual);
	}
}

#[no_mangle]
pub unsafe extern fn classification_compute(weights: *mut c_void, x: f64, y: f64) -> i32 {
	let weights = &mut *(weights as *mut Vec<f64>);
	classification_sign(weights, x, y)
}

fn classification_update_weights(weights: &mut Vec<f64>, x: f64, y: f64, expected: i32, actual: i32) {
	let alpha = 0.1;
	let diff = (expected - actual) as f64;
	
	weights[0] = weights[0] + alpha * diff * 1.;
	weights[1] = weights[1] + alpha * diff * x;
	weights[2] = weights[2] + alpha * diff * y;
}

fn classification_sign(weights: &Vec<f64>, x: f64, y: f64) -> i32 {
	match (weights[0] + x * weights[1] + y * weights[2]) > 0. {
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