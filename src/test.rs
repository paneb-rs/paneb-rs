use super::model::Model;

pub struct TestModel {}

impl TestModel {
	pub fn new() -> Self {
		TestModel {}
	}
}

impl Model<f64, f64> for TestModel {
	fn train(&self, _input: f64, _output: f64) {}
	
	fn compute(&self, input: f64) -> f64 {
		input + 42.
	}
}

#[no_mangle]
pub unsafe extern fn test_model_create() -> *const TestModel {
	&TestModel::new() as *const TestModel
}

#[no_mangle]
pub unsafe extern fn test_model_compute(model: *const TestModel, input: f64) -> f64 {
	(*model).compute(input)
}

#[cfg(test)]
mod test {
	use ::model::Model;
	use super::TestModel;
	
	#[test]
	fn should_add_42_to_input() {
		let input = 7.;
		let expected = 49.;
		
		let model = TestModel::new();
		let output = model.compute(input);
		
		assert_eq!(expected, output);
	}
}