use super::model::Model;

pub struct ClassificationModel {

}

impl ClassificationModel {
	pub fn new() -> Self {
		ClassificationModel {}
	}
}

impl Model<f64, bool> for ClassificationModel {
	
	fn train(&self, _input: f64, _output: bool) {
		unimplemented!()
	}
	
	fn compute(&self, _input: f64) -> bool {
		unimplemented!()
	}
}