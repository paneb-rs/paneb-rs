extern crate nalgebra;

#[no_mangle]
pub extern fn add(i: u8, j: u8) -> u8 {
	i + j
}

#[cfg(test)]
mod test {
	use add;
	
	#[test]
	fn test() {
		assert_eq!(2, add(1, 1));
	}
}