use std::os::raw::c_void;
use std::slice::from_raw_parts;

use nalgebra::{DMatrix};

#[no_mangle]
pub unsafe extern fn regression_create(rows: i32, columns: i32, raw_inputs: *mut c_void) -> *mut c_void {
	let inputs = from_raw_parts(raw_inputs as *mut f64, (rows * columns) as usize);
	
	Box::into_raw(Box::new(
		DMatrix::from_column_slice(rows as usize, columns as usize, inputs)
	)) as *mut c_void
}

#[no_mangle]
pub unsafe extern fn regression_train(raw_matrix: *mut c_void, rows: i32, columns: i32, raw_outputs: *mut c_void) -> *mut c_void {
	let input_matrix = &*(raw_matrix as *mut DMatrix<f64>);
	
	let outputs = from_raw_parts(raw_outputs as *mut f64, (rows * columns) as usize);
	let output_matrix = DMatrix::from_column_slice(rows as usize, columns as usize, outputs);
	
	let transposed_inputs = input_matrix.transpose();
	let multiplied_inputs = input_matrix * transposed_inputs.clone();
	
	let inversed_inputs = multiplied_inputs.try_inverse().unwrap();
	let final_inputs = transposed_inputs * inversed_inputs;
	
	let result = output_matrix * final_inputs;
	
	//Box::into_raw(Box::new(result)) as *mut c_void
	raw_matrix
}