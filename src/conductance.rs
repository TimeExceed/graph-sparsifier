use crate::cut;
use algograph::graph::*;
use std::{cmp::min, collections::HashSet};

pub fn conductance<G: QueryableGraph>(graph: &G, s: &HashSet<VertexId, ahash::RandomState>) -> f64 {
    assert!(!s.is_empty());
    assert!(s.len() < graph.vertex_size());
    let cut = cut(graph, s) as f64;
    let vol_s = volumn(graph, s);
    let vol_g = graph.edge_size() * 2;
    let divider = min(vol_s, vol_g - vol_s) as f64;
    cut / divider
}

pub fn volumn<G: QueryableGraph>(graph: &G, s: &HashSet<VertexId, ahash::RandomState>) -> usize {
    let mut res = 0;
    for u in s.iter() {
        res += graph.out_edges(u).count();
    }
    res
}

#[cfg(test)]
pub fn min_conductance<G: QueryableGraph>(graph: &G) -> Option<f64> {
    let vertices: Vec<_> = graph.iter_vertices().collect();
    let mut min_conductance = None;
    for vs in crate::subset::Subsets::new(vertices.iter()) {
        if vs.is_empty() || vs.len() == vertices.len() {
            continue;
        }
        let vs: HashSet<_, ahash::RandomState> = vs.into_iter().copied().collect();
        let conductance = conductance(graph, &vs);
        if let Some(ref mut min_c) = min_conductance {
            if *min_c > conductance {
                *min_c = conductance;
            }
        } else {
            min_conductance = Some(conductance);
        }
    }
    min_conductance
}
