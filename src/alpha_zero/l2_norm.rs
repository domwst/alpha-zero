use tch::{
    nn::{Conv, ConvTransposeND, Linear},
    Tensor,
};

pub trait L2Norm {
    fn l2(&self) -> Tensor;
}

impl<T: L2Norm> L2Norm for &T {
    fn l2(&self) -> Tensor {
        (*self).l2()
    }
}

impl L2Norm for Tensor {
    fn l2(&self) -> Tensor {
        (self * self).sum(None)
    }
}

impl L2Norm for Linear {
    fn l2(&self) -> Tensor {
        self.ws.l2()
    }
}

impl<ND> L2Norm for Conv<ND> {
    fn l2(&self) -> Tensor {
        self.ws.l2()
    }
}

impl<ND> L2Norm for ConvTransposeND<ND> {
    fn l2(&self) -> Tensor {
        self.ws.l2()
    }
}
