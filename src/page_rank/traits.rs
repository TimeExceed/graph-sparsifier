use algograph::graph::*;
use std::collections::HashMap;

pub trait PageRank {
    type Result: PageRankResult;

    fn calc(&self, start: &HashMap<VertexId, f64, ahash::RandomState>) -> Self::Result;
}

pub trait PageRankResult {
    fn page_rank(&self) -> &HashMap<VertexId, f64, ahash::RandomState>;
    fn debug<'a, G: QueryableGraph>(&'a self, graph: &'a G) -> impl std::fmt::Debug + 'a;
}
