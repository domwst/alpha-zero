use tch::Tensor;

pub trait AlphaZeroNet {
    fn forward_t(&self, xs: &Tensor, is_training: bool) -> (Tensor, Tensor);
}
