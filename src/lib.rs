extern crate nalgebra;

#[no_mangle]
pub extern fn super_add(value: f64) -> f64 {
	value + 42.
}

#[cfg(test)]
mod test {
	use super_add;
	
	#[test]
	fn test() {
		assert_eq!(49., super_add(7.));
	}
}