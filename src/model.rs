pub trait Model<Input, Output> {
	fn train(&self, input: Input, output: Output);
	fn compute(&self, input: Input) -> Output;
}
