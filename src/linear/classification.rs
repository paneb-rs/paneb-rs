use rand;
use rand::Rng;

#[no_mangle]
pub unsafe extern fn classification_create() -> *mut [f64; 3] {
	let mut random = rand::thread_rng();
	
	Box::into_raw(Box::new([
		random.gen_range(-1., 1.),
		random.gen_range(-1., 1.),
		random.gen_range(-1., 1.)
	]))
}

#[no_mangle]
pub unsafe extern fn classification_weights(weights: *mut [f64; 3], index: i32) -> f64 {
	(*weights)[index as usize]
}

#[no_mangle]
pub unsafe extern fn classification_train(weights: *mut [f64; 3], x: f64, y: f64, expected: i32) -> *mut [f64; 3] {
	let actual = compute_actual(&(*weights), x, y);
	
	if expected != actual {
		update_weights(&mut (*weights), x, y, expected, actual);
	}
	
	weights
}

fn update_weights(weights: &mut [f64; 3], x: f64, y: f64, expected: i32, actual: i32) {
	let alpha = 0.1;
	let diff = (expected - actual) as f64;
	
	weights[0] = weights[0] + alpha * diff * 1.;
	weights[1] = weights[1] + alpha * diff * x;
	weights[2] = weights[2] + alpha * diff * y;
}

fn compute_actual(weights: &[f64; 3], x: f64, y: f64) -> i32 {
	match (weights[0] + x * weights[1] + y * weights[2]) > 0. {
		true => 1,
		false => -1
	}
}