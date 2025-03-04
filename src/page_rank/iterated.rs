use super::*;
use crate::*;
use algograph::graph::{QueryableGraph, VertexId};
use std::collections::{BTreeMap, HashMap};

pub struct IteratedPageRank<'a, G>
where
    G: QueryableGraph,
{
    graph: &'a G,
    damping: f64,
    epsilon: f64,
    transitions: BTreeMap<(VertexId, VertexId), f64>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub damping: f64,
    pub epsilon: f64,
}

#[derive(Debug, Clone)]
pub struct Result {
    pub page_rank: HashMap<VertexId, f64, ahash::RandomState>,
    pub delta: HashMap<VertexId, f64, ahash::RandomState>,
}

impl<'a, G: QueryableGraph> IteratedPageRank<'a, G> {
    pub fn new(g: &'a G, config: &Config) -> Self {
        let damping = config.damping;
        assert!((0.0..=1.0).contains(&damping), "damping={damping}");
        let epsilon = config.epsilon;
        assert!(epsilon > 0.0, "epsilon={epsilon}");
        let transitions = {
            let mut transitions = BTreeMap::new();
            for u in g.iter_vertices() {
                let n = g.out_edges(&u).count() + 1;
                let unit = damping / (n as f64);
                for v in g.out_edges(&u).map(|e| e.sink).chain([u]) {
                    if let Some(w) = transitions.get_mut(&(u, v)) {
                        *w += unit;
                    } else {
                        transitions.insert((u, v), unit);
                    }
                }
            }
            transitions
        };
        Self {
            graph: g,
            damping,
            epsilon,
            transitions,
        }
    }
}

impl<G: QueryableGraph> PageRank for IteratedPageRank<'_, G> {
    type Result = self::Result;

    fn calc(&self, start: &HashMap<VertexId, f64, ahash::RandomState>) -> Self::Result {
        let damping = self.damping;
        let epsilon = self.epsilon;
        let mut p = {
            let mut p = HashMap::with_hasher(ahash::RandomState::new());
            let mut p_sum = 0.0;
            for v in self.graph.iter_vertices() {
                if let Some(w) = start.get(&v) {
                    p.insert(v, *w);
                    p_sum += w;
                } else {
                    p.insert(v, 0.0);
                }
            }
            assert!((p_sum - 1.0).abs() < 1e-7, "p_sum={p_sum}");
            p
        };
        let mut r = HashMap::with_hasher(ahash::RandomState::new());
        let mut delta = HashMap::with_hasher(ahash::RandomState::new());
        loop {
            for v in self.graph.iter_vertices() {
                if let Some(w) = start.get(&v) {
                    r.insert(v, *w * (1.0 - damping));
                } else {
                    r.insert(v, 0.0);
                }
            }
            for ((v0, v1), w) in self.transitions.iter() {
                let from = p.get(v0).unwrap();
                let to = r.get_mut(v1).unwrap();
                *to += from * w;
            }

            delta.clear();
            for v in self.graph.iter_vertices() {
                let a = p.get(&v).unwrap();
                let b = r.get(&v).unwrap();
                delta.insert(v, a - b);
            }

            if norm_1(&delta) < epsilon {
                return Self::Result {
                    page_rank: r,
                    delta,
                };
            }

            std::mem::swap(&mut p, &mut r);
            r.clear();
        }
    }
}

impl PageRankResult for self::Result {
    fn page_rank(&self) -> &HashMap<VertexId, f64, ahash::RandomState> {
        &self.page_rank
    }

    fn debug<'a, G: QueryableGraph>(&'a self, graph: &'a G) -> impl std::fmt::Debug + 'a {
        ResultDebug {
            graph,
            result: self,
        }
    }
}

pub struct ResultDebug<'a, G: QueryableGraph> {
    graph: &'a G,
    result: &'a self::Result,
}

impl<G: QueryableGraph> std::fmt::Debug for ResultDebug<'_, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for v in self.graph.iter_vertices() {
            let p = self.result.page_rank.get(&v).unwrap();
            let d = self.result.delta.get(&v).unwrap();
            writeln!(f, "{v:?}: {p:?}, {d:?}")?;
        }
        Ok(())
    }
}
