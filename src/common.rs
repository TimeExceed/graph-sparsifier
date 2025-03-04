use algograph::graph::*;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

pub fn norm_1<K: Ord + Hash>(v: &HashMap<K, f64, ahash::RandomState>) -> f64 {
    v.values().map(|x| x.abs()).sum()
}

pub fn support(
    p: &HashMap<VertexId, f64, ahash::RandomState>,
) -> HashSet<VertexId, ahash::RandomState> {
    p.iter()
        .filter_map(|(vert, val)| {
            if *val < -1e-7 || *val > 1e-7 {
                Some(*vert)
            } else {
                None
            }
        })
        .collect()
}
