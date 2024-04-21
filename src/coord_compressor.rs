use std::{collections::HashMap, hash::Hash};

pub struct CoordCompressor<T: Hash + Eq + Clone> {
    fwd: HashMap<T, usize>,
    back: Vec<T>,
}

impl<T: Hash + Eq + Clone> CoordCompressor<T> {
    pub fn new() -> Self {
        Self {
            fwd: HashMap::new(),
            back: Vec::new(),
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            fwd: HashMap::with_capacity(n),
            back: Vec::with_capacity(n),
        }
    }

    pub fn compress(&mut self, v: &T) -> usize {
        let (_, v) = self.fwd.raw_entry_mut().from_key(v).or_insert_with(|| {
            let idx = self.back.len();
            self.back.push(v.clone());
            (v.clone(), idx)
        });
        *v
    }

    pub fn decompress(&self, idx: usize) -> &T {
        &self.back[idx]
    }
}
