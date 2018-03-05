extern crate nalgebra;

pub trait Model<Input, Output> {
	fn train(&self, input: Input, output: Output);
	fn compute(&self, input: Input) -> Output;
}

pub struct TestModel {}

impl TestModel {
	
	pub fn new() -> Self {
		TestModel{}
	}
}

impl Model<f64, f64> for TestModel {
	
	fn train(&self, _input: f64, _output: f64) {
		// TODO: train model
	}
	
	fn compute(&self, input: f64) -> f64 {
		input + 40.
	}
}

#[no_mangle]
pub unsafe extern fn test_model_create() -> *const TestModel {
	&TestModel::new() as *const TestModel
}

#[no_mangle]
pub unsafe extern fn test_model_compute(model: *const TestModel, input: f64) -> f64{
	(*model).compute(input)
}

#[cfg(test)]
mod test {

}