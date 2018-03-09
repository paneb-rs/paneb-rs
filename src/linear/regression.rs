use std::os::raw::c_void;
use std::slice::from_raw_parts;

use nalgebra::DMatrix;

#[no_mangle]
pub unsafe extern fn regression_compute(
	input_rows: i32, input_cols: i32, input_raw: *mut c_void,
	output_rows: i32, output_cols: i32, output_raw: *mut c_void
) -> *mut c_void {
	let inputs = from_raw_parts(input_raw as *mut f64, (input_rows * input_cols) as usize);
	let input_matrix = DMatrix::from_row_slice(input_rows as usize, input_cols as usize, inputs);
	
	let outputs = from_raw_parts(output_raw as *mut f64, (output_rows * output_cols) as usize);
	let output_matrix = DMatrix::from_row_slice(output_rows as usize, output_cols as usize, outputs);
	
	let transposed_inputs = input_matrix.transpose();
	let multiplied_inputs = transposed_inputs.clone() * input_matrix;
	let inversed_inputs = multiplied_inputs.pseudo_inverse(10E-49);
	let final_inputs = inversed_inputs * transposed_inputs;
	let result = final_inputs * output_matrix;
	
	Box::into_raw(Box::new(result.data)) as *mut c_void
}

#[no_mangle]
pub unsafe extern fn regression_point(weights_raw: *mut c_void, inputs_size: i32, inputs_raw: *mut c_void) -> f64 {
	let weights = &*(weights_raw as *mut Box<[f64]>);
	let inputs = from_raw_parts(inputs_raw as *mut f64, inputs_size as usize);
	
	let mut acc = weights[0];
	
	for index in 0..inputs.len() {
		acc += inputs[index] * weights[index + 1];
	}
	
	acc
}

#[cfg(test)]
mod test {
	use std::os::raw::c_void;
	use super::*;
	
	#[test]
	fn should_regress() {
		let mut raw_inputs = vec![
			1., 1., 8.,
			1., 1., -2.,
			1., 1., -2.,
			1., -2., -1.
		];
		let inputs = raw_inputs.as_mut_ptr() as *mut c_void;
		
		let mut raw_outputs = vec![
			1.,
			1.,
			-1.,
			-1.
		];
		let outputs = raw_outputs.as_mut_ptr() as *mut c_void;
		
		unsafe {
			let model = regression_compute(
				4, 3, inputs,
				4, 1, outputs
			);
			regression_point(model, 2, outputs);
		}
	}
}