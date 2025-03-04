use algograph::graph::*;
use std::{cmp::min, collections::HashSet};

pub fn cheeger<G: QueryableGraph>(graph: &G, s: &HashSet<VertexId, ahash::RandomState>) -> f64 {
    assert!(!s.is_empty());
    assert!(s.len() < graph.vertex_size());
    let cut = cut(graph, s) as f64;
    let divider = min(s.len(), graph.vertex_size() - s.len()) as f64;
    cut / divider
}

pub fn cut<G: QueryableGraph>(graph: &G, s: &HashSet<VertexId, ahash::RandomState>) -> usize {
    let mut res = 0;
    for u in s.iter() {
        for e in graph.out_edges(u) {
            let v = e.sink;
            if !s.contains(&v) {
                res += 1;
            }
        }
    }
    res
}

#[cfg(test)]
pub fn min_cheeger<G: QueryableGraph>(graph: &G) -> Option<f64> {
    let vertices: Vec<_> = graph.iter_vertices().collect();
    let mut min_cheeger = None;
    for vs in crate::subset::Subsets::new(vertices.iter()) {
        if vs.is_empty() || vs.len() == vertices.len() {
            continue;
        }
        let vs: HashSet<_, ahash::RandomState> = vs.into_iter().copied().collect();
        let cheeger = cheeger(graph, &vs);
        if let Some(ref mut min_c) = min_cheeger {
            if *min_c > cheeger {
                *min_c = cheeger;
            }
        } else {
            min_cheeger = Some(cheeger);
        }
    }
    min_cheeger
}
