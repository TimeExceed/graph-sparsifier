use super::*;
use algograph::graph::*;
use keyed_priority_queue::KeyedPriorityQueue;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    iter::Iterator,
};

pub struct ApproxPageRank<'a, G>
where
    G: QueryableGraph,
{
    graph: &'a G,
    alpha: f64,
    epsilon: f64,
    degrees: HashMap<VertexId, f64, ahash::RandomState>,
    transitions: HashMap<VertexId, Vec<(VertexId, f64)>, ahash::RandomState>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub damping: f64,
    pub epsilon: f64,
}

#[derive(Debug, Clone)]
pub struct Result {
    pub page_rank: HashMap<VertexId, f64, ahash::RandomState>,
    pub r: HashMap<VertexId, f64, ahash::RandomState>,
}

impl<'a, G: QueryableGraph> ApproxPageRank<'a, G> {
    pub fn new(g: &'a G, config: &Config) -> Self {
        assert!(
            (0.0..=1.0).contains(&config.damping),
            "damping={}",
            config.damping
        );
        assert!(config.epsilon > 0.0, "epsilon={}", config.epsilon);
        assert!(g.vertex_size() > 0, "vertex size={}", g.vertex_size());

        let alpha = 1.0 - config.damping;
        let degrees = {
            // `+1` for making the graph self-looped
            let mut degrees: HashMap<_, _, ahash::RandomState> =
                g.iter_vertices().map(|u| (u, 1.0)).collect();
            for e in g.iter_edges() {
                if e.source == e.sink {
                    *degrees.get_mut(&e.source).unwrap() += 1.0;
                } else {
                    *degrees.get_mut(&e.source).unwrap() += 1.0;
                    *degrees.get_mut(&e.sink).unwrap() += 1.0;
                };
            }
            degrees
        };
        let epsilon = {
            let max_d = degrees.values().max_by_key(|w| FullOrdFloat(**w)).unwrap();
            config.epsilon / (g.vertex_size() as f64) / max_d
        };
        let unit = |u: VertexId| -> f64 {
            let d_u = *degrees.get(&u).unwrap();
            (1.0 - alpha) / d_u / 2.0
        };
        let transitions: HashMap<VertexId, Vec<(VertexId, f64)>, ahash::RandomState> = {
            let mut transitions: HashMap<_, _, ahash::RandomState> = g
                .iter_vertices()
                .map(|u| {
                    let val: BTreeMap<_, _> = [(u, unit(u))].into_iter().collect();
                    (u, val)
                })
                .collect();
            for e in g.iter_edges() {
                let vs: &[(VertexId, VertexId)] = if e.source == e.sink {
                    &[(e.source, e.sink)]
                } else {
                    let u = e.source;
                    let v = e.sink;
                    &[(u, v), (v, u)]
                };
                for (u, v) in vs.iter() {
                    let sinks = transitions.get_mut(u).unwrap();
                    if let Some(w) = sinks.get_mut(v) {
                        *w += unit(*u);
                    } else {
                        sinks.insert(*v, unit(*u));
                    }
                }
            }
            transitions
                .into_iter()
                .map(|(u, vs)| {
                    let vs: Vec<_> = vs.into_iter().collect();
                    (u, vs)
                })
                .collect()
        };
        Self {
            graph: g,
            alpha,
            epsilon,
            degrees,
            transitions,
        }
    }
}

impl<G: QueryableGraph> PageRank for ApproxPageRank<'_, G> {
    type Result = self::Result;

    fn calc(&self, start: &HashMap<VertexId, f64, ahash::RandomState>) -> Self::Result {
        let mut p = {
            let mut p = HashMap::with_hasher(ahash::RandomState::new());
            for v in self.graph.iter_vertices() {
                p.insert(v, 0.0);
            }
            p
        };
        let mut r = {
            let mut r = HashMap::with_hasher(ahash::RandomState::new());
            for v in self.graph.iter_vertices() {
                if let Some(w) = start.get(&v) {
                    r.insert(v, *w);
                } else {
                    r.insert(v, 0.0);
                }
            }
            r
        };
        let mut q = {
            let mut q = KeyedPriorityQueue::new();
            for v in self.graph.iter_vertices() {
                if let Some(x) = self.exceeded(v, &r) {
                    q.push(v, FullOrdFloat(x));
                }
            }
            q
        };
        while let Some((u, _)) = q.pop() {
            self.push(u, &mut p, &mut r);

            if let Some(xs) = self.transitions.get(&u) {
                for (v, _) in xs.iter() {
                    if let Some(x) = self.exceeded(*v, &r) {
                        q.push(*v, FullOrdFloat(x));
                    } else {
                        q.remove(v);
                    }
                }
            }
        }
        Self::Result { page_rank: p, r }
    }
}

impl<G: QueryableGraph> ApproxPageRank<'_, G> {
    fn push(
        &self,
        u: VertexId,
        p: &mut HashMap<VertexId, f64, ahash::RandomState>,
        r: &mut HashMap<VertexId, f64, ahash::RandomState>,
    ) {
        let r_u = *r.get(&u).unwrap();
        *p.get_mut(&u).unwrap() += self.alpha * r_u;
        *r.get_mut(&u).unwrap() = r_u * (1.0 - self.alpha) / 2.0;

        if let Some(xs) = self.transitions.get(&u) {
            for (v, w) in xs.iter() {
                *r.get_mut(v).unwrap() += r_u * w;
            }
        }
    }

    fn exceeded(&self, v: VertexId, r: &HashMap<VertexId, f64, ahash::RandomState>) -> Option<f64> {
        let r_v = *r.get(&v).unwrap();
        let d_v = *self.degrees.get(&v).unwrap();
        if r_v > self.epsilon * d_v {
            Some(r_v - self.epsilon * d_v)
        } else {
            None
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
            let r = self.result.r.get(&v).unwrap();
            writeln!(f, "{v:?}: {p:?}, {r:?}")?;
        }
        Ok(())
    }
}

struct FullOrdFloat(f64);
impl PartialOrd for FullOrdFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FullOrdFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}
impl PartialEq for FullOrdFloat {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for FullOrdFloat {}

fn harmonic_number(n: usize) -> f64 {
    // https://en.wikipedia.org/wiki/Harmonic_number
    const EULER: f64 = 0.57721_56649_01532;
    let x = n as f64;
    x.ln() * EULER
}

pub struct Sparsifier<'a, G>
where
    G: QueryableGraph + VertexShrinkableGraph + Clone,
{
    graph: &'a G,
    cur_graph: AdaptiveOwnedGraph<'a, G>,
    last_cached: RefCell<Cached>,
}

#[derive(Default)]
struct Cached {
    cached_vertex_size: usize,
    start_candidates: Vec<VertexId>,
}

enum AdaptiveOwnedGraph<'a, G>
where
    G: QueryableGraph + VertexShrinkableGraph + EdgeShrinkableGraph,
{
    Borrow(ShadowedSubgraph<'a, G>),
    Owned(G),
}

impl<'a, G> Sparsifier<'a, G>
where
    G: QueryableGraph + VertexShrinkableGraph + EdgeShrinkableGraph + DirectedOrNot + Clone,
{
    pub fn new(graph: &'a G) -> Self {
        assert!(!G::DIRECTED_OR_NOT);
        Self {
            graph,
            cur_graph: AdaptiveOwnedGraph::Borrow(ShadowedSubgraph::new(graph)),
            last_cached: RefCell::new(Cached::default()),
        }
    }

    fn start_vertex(
        &self,
    ) -> (
        VertexId,
        Option<ApproxPageRank<'_, AdaptiveOwnedGraph<'_, G>>>,
    ) {
        assert!(
            self.cur_graph.vertex_size() > 0,
            "# of vertex: {}",
            self.cur_graph.vertex_size()
        );
        let mut cached = self.last_cached.borrow_mut();
        let should_calibrate = cached.start_candidates.is_empty()
            || self.cur_graph.vertex_size() * 3 < cached.cached_vertex_size * 2;
        let apr = if should_calibrate {
            assert!(self.cur_graph.vertex_size() > 0);
            let start: HashMap<_, _, ahash::RandomState> = {
                let n = self.cur_graph.vertex_size() as f64;
                self.cur_graph
                    .iter_vertices()
                    .map(|v| (v, 1.0 / n))
                    .collect()
            };
            let apr = new_apr(&self.cur_graph);
            let result = apr.calc(&start);
            cached.start_candidates.clear();
            for v in self.cur_graph.iter_vertices() {
                cached.start_candidates.push(v);
            }
            cached
                .start_candidates
                .sort_by_key(|v| FullOrdFloat(*result.page_rank.get(v).unwrap()));
            cached.cached_vertex_size = self.cur_graph.vertex_size();
            Some(apr)
        } else {
            None
        };
        let mut start = None;
        while let Some(v) = cached.start_candidates.pop() {
            if self.cur_graph.contains_vertex(&v) {
                start = Some(v);
                break;
            }
        }
        (start.unwrap(), apr)
    }

    fn expander<GG>(
        &self,
        apr: &ApproxPageRank<'_, GG>,
        start_vertex: VertexId,
    ) -> HashSet<VertexId, ahash::RandomState>
    where
        GG: QueryableGraph,
    {
        let g = &self.cur_graph;
        if g.vertex_size() == 1 {
            return g.iter_vertices().collect();
        }
        let pr = {
            let start = {
                let mut start = HashMap::with_hasher(ahash::RandomState::new());
                start.insert(start_vertex, 1.0);
                start
            };
            let res = apr.calc(&start);
            res.page_rank
        };
        let vertices = {
            let mut vs: Vec<_> = g.iter_vertices().collect();
            vs.sort_by(|a, b| {
                let p_a = pr.get(a).unwrap();
                let p_b = pr.get(b).unwrap();
                p_a.partial_cmp(p_b).unwrap().reverse()
            });
            vs
        };
        let mut vert_iter = vertices.iter().copied();
        let s = vert_iter.next().unwrap();
        let mut res: HashSet<_, ahash::RandomState> = [s].into_iter().collect();
        for v in vert_iter {
            /*
            Here we have to answer "is the vertex 'v' high cohesion to the
            existant vertex set 'res'"?
            Practically, we count the number of edges from 'v' to 'res'.
            We say "yes", when the number is greater than sqrt(|res|).
             */
            let involved = g.out_edges(&v).filter(|e| res.contains(&e.sink)).count();
            if involved * involved < res.len() {
                break;
            }
            res.insert(v);
        }
        res
    }

    fn postprocess(
        &mut self,
        expander: &HashSet<VertexId, ahash::RandomState>,
    ) -> SelectedSubgraph<'a, G> {
        for v in expander.iter() {
            let _ = self.cur_graph.remove_vertex(v);
        }
        if self.cur_graph.vertex_size() * 2 <= self.graph.vertex_size() {
            self.own_cur_graph();
        }

        let mut res = SelectedSubgraph::new(self.graph);
        for u in expander.iter() {
            res.disclose_vertex(*u);
        }
        for u in expander.iter() {
            for e in self.graph.out_edges(u) {
                if expander.contains(&e.sink) {
                    res.disclose_edge(e.id);
                }
            }
        }
        res
    }

    fn own_cur_graph(&mut self) {
        let AdaptiveOwnedGraph::Borrow(shadowed) = &self.cur_graph else {
            return;
        };
        let mut cloned = self.graph.clone();
        for u in self.graph.iter_vertices() {
            if !shadowed.contains_vertex(&u) {
                let _ = cloned.remove_vertex(&u);
            }
        }
        self.cur_graph = AdaptiveOwnedGraph::Owned(cloned);
    }
}

fn new_apr<G>(g: &G) -> ApproxPageRank<'_, G>
where
    G: QueryableGraph,
{
    // On the one hand, let us detect expanders whose diameter is $n$,
    // i.e., arbitrary two vertices in the expander are connected within $n$ edges.
    // That is to say, `damping`'s $n$-th square should be small enough.
    // For example, if we want expanders whose diameter is 4, and we
    // consider 0.1 as small enough, then we will have `damping` must >=0.56.
    //
    // On the other hand, small `damping`` causes big differences between
    // results of the iterative method and those of the approximated method.
    // We examined it over our unit tests, we found the following looks good.
    let damping = 0.9;
    let vol_total = g.edge_size() * 2;
    let gamma = harmonic_number(vol_total);
    let epsilon = 1.0 / (2.0 * gamma);
    let cfg = Config { damping, epsilon };
    ApproxPageRank::new(g, &cfg)
}

impl<'a, G> Iterator for Sparsifier<'a, G>
where
    G: QueryableGraph + VertexShrinkableGraph + EdgeShrinkableGraph + DirectedOrNot + Clone,
{
    type Item = SelectedSubgraph<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_graph.vertex_size() == 0 {
            return None;
        }
        if self.cur_graph.vertex_size() == 1 {
            let u = self.cur_graph.iter_vertices().next().unwrap();
            let expander: HashSet<_, ahash::RandomState> = [u].into_iter().collect();
            return Some(self.postprocess(&expander));
        }
        let expander: HashSet<_, ahash::RandomState> = if self.cur_graph.edge_size() == 0 {
            let v = self.cur_graph.iter_vertices().next().unwrap();
            [v].into_iter().collect()
        } else {
            let (start_vertex, apr) = self.start_vertex();
            if let Some(apr) = apr {
                self.expander(&apr, start_vertex)
            } else {
                let apr = new_apr(&self.cur_graph);
                self.expander(&apr, start_vertex)
            }
        };

        Some(self.postprocess(&expander))
    }
}

impl<G> QueryableGraph for AdaptiveOwnedGraph<'_, G>
where
    G: QueryableGraph + VertexShrinkableGraph + EdgeShrinkableGraph,
{
    fn vertex_size(&self) -> usize {
        match self {
            Self::Borrow(x) => x.vertex_size(),
            Self::Owned(x) => x.vertex_size(),
        }
    }

    fn iter_vertices(&self) -> Box<dyn Iterator<Item = VertexId> + '_> {
        match self {
            Self::Borrow(x) => x.iter_vertices(),
            Self::Owned(x) => x.iter_vertices(),
        }
    }

    fn contains_vertex(&self, v: &VertexId) -> bool {
        match self {
            Self::Borrow(x) => x.contains_vertex(v),
            Self::Owned(x) => x.contains_vertex(v),
        }
    }

    fn edge_size(&self) -> usize {
        match self {
            Self::Borrow(x) => x.edge_size(),
            Self::Owned(x) => x.edge_size(),
        }
    }

    fn iter_edges(&self) -> Box<dyn Iterator<Item = Edge> + '_> {
        match self {
            Self::Borrow(x) => x.iter_edges(),
            Self::Owned(x) => x.iter_edges(),
        }
    }

    fn contains_edge(&self, e: &EdgeId) -> bool {
        match self {
            Self::Borrow(x) => x.contains_edge(e),
            Self::Owned(x) => x.contains_edge(e),
        }
    }

    fn find_edge(&self, e: &EdgeId) -> Option<Edge> {
        match self {
            Self::Borrow(x) => x.find_edge(e),
            Self::Owned(x) => x.find_edge(e),
        }
    }

    fn edges_connecting(
        &self,
        source: &VertexId,
        sink: &VertexId,
    ) -> Box<dyn Iterator<Item = Edge> + '_> {
        match self {
            Self::Borrow(x) => x.edges_connecting(source, sink),
            Self::Owned(x) => x.edges_connecting(source, sink),
        }
    }

    fn in_edges(&self, v: &VertexId) -> Box<dyn Iterator<Item = Edge> + '_> {
        match self {
            Self::Borrow(x) => x.in_edges(v),
            Self::Owned(x) => x.in_edges(v),
        }
    }

    fn out_edges(&self, v: &VertexId) -> Box<dyn Iterator<Item = Edge> + '_> {
        match self {
            Self::Borrow(x) => x.out_edges(v),
            Self::Owned(x) => x.out_edges(v),
        }
    }
}

impl<G> VertexShrinkableGraph for AdaptiveOwnedGraph<'_, G>
where
    G: QueryableGraph + VertexShrinkableGraph + EdgeShrinkableGraph,
{
    fn remove_vertex(&mut self, vertex: &VertexId) -> Box<dyn Iterator<Item = Edge> + 'static> {
        match self {
            Self::Borrow(x) => x.remove_vertex(vertex),
            Self::Owned(x) => x.remove_vertex(vertex),
        }
    }
}

impl<G> EdgeShrinkableGraph for AdaptiveOwnedGraph<'_, G>
where
    G: QueryableGraph + VertexShrinkableGraph + EdgeShrinkableGraph,
{
    fn remove_edge(&mut self, edge: &EdgeId) -> Option<Edge> {
        match self {
            Self::Borrow(x) => x.remove_edge(edge),
            Self::Owned(x) => x.remove_edge(edge),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    #[test]
    fn dumbbell_0() {
        const N: usize = 5;

        let mut g = undirected::TreeBackedGraph::new();
        let v0 = *add_complete_graph(&mut g, N).first().unwrap();
        let v1 = *add_complete_graph(&mut g, N).first().unwrap();
        g.add_edge(v0, v1);

        let mut sparsified = Sparsifier::new(&g);
        assert_eq!(sparsified.next().unwrap().vertex_size(), N);
        assert_eq!(sparsified.next().unwrap().vertex_size(), N);
    }

    #[test]
    fn dumbbell_1() {
        const N: usize = 4;

        let mut g = undirected::TreeBackedGraph::new();
        let v0 = *add_complete_graph(&mut g, N).first().unwrap();
        let v1 = *add_complete_graph(&mut g, N).first().unwrap();
        let vm = g.add_vertex();
        g.add_edge(v0, vm);
        g.add_edge(v1, vm);

        let mut sparsified = Sparsifier::new(&g);
        assert_eq!(sparsified.next().unwrap().vertex_size(), N);
        assert_eq!(sparsified.next().unwrap().vertex_size(), N);
        assert_eq!(sparsified.next().unwrap().vertex_size(), 1);
    }

    #[test]
    fn polliwog_4head_1tail() {
        const H: usize = 4;
        const T: usize = 1;

        let mut g = undirected::TreeBackedGraph::new();
        let mut v0 = *add_complete_graph(&mut g, H).first().unwrap();
        for _ in 0..T {
            let v = g.add_vertex();
            g.add_edge(v0, v);
            v0 = v;
        }

        let sparsified = Sparsifier::new(&g);
        let trial: Vec<_> = sparsified.map(|subg| subg.vertex_size()).collect();
        let oracle = {
            let mut oracle = vec![H];
            oracle.resize(T + 1, 1);
            oracle
        };
        assert_eq!(trial, oracle);
    }

    #[test]
    fn rope_10() {
        const N: usize = 10;

        let mut g = undirected::TreeBackedGraph::new();
        let mut u = g.add_vertex();
        for _ in 0..N {
            let v = g.add_vertex();
            g.add_edge(u, v);
            u = v;
        }

        let sparsified = Sparsifier::new(&g);
        let trial: Vec<_> = sparsified.map(|subg| subg.vertex_size()).collect();
        trial.iter().for_each(|vs| assert!(*vs <= 2));
        assert!(trial.iter().filter(|vs| **vs == 1).count() * 2 < trial.len());
    }

    #[test]
    fn grape4_2() {
        const N: usize = 4;
        const M: usize = 2;

        let mut g = undirected::TreeBackedGraph::new();
        let mut vs = add_complete_graph(&mut g, N);
        for _ in 1..M {
            let tail = vs.pop().unwrap();
            vs = add_complete_graph(&mut g, N);
            let head = *vs.first().unwrap();
            g.add_edge(tail, head);
        }

        let sparsified = Sparsifier::new(&g);
        let trial: Vec<_> = sparsified.map(|subg| subg.vertex_size()).collect();
        let oracle: Vec<_> = (0..M).map(|_| N).collect();
        assert_eq!(trial, oracle);
    }

    #[test]
    fn grape4_3() {
        const N: usize = 4;
        const M: usize = 3;

        let mut g = undirected::TreeBackedGraph::new();
        let mut vs = add_complete_graph(&mut g, N);
        for _ in 1..M {
            let tail = vs.pop().unwrap();
            vs = add_complete_graph(&mut g, N);
            let head = *vs.first().unwrap();
            g.add_edge(tail, head);
        }

        let sparsified = Sparsifier::new(&g);
        let trial: Vec<_> = sparsified.map(|subg| subg.vertex_size()).collect();
        let oracle: Vec<_> = (0..M).map(|_| N).collect();
        assert_eq!(trial, oracle);
    }

    fn add_complete_graph<G: GrowableGraph>(g: &mut G, n: usize) -> Vec<VertexId> {
        assert!(n > 0, "{n}");
        let v0 = g.add_vertex();
        let mut vs = vec![v0];
        for _ in 1..n {
            let v = g.add_vertex();
            vs.push(v);
        }
        let mut it0 = vs.iter();
        while let Some(v1) = it0.next() {
            for v2 in it0.clone() {
                g.add_edge(*v1, *v2);
            }
        }
        vs
    }

    #[test]
    fn disconnect() {
        const N: usize = 5;

        let mut g = undirected::TreeBackedGraph::new();
        let _ = add_complete_graph(&mut g, N);
        let _ = add_complete_graph(&mut g, N);

        let mut sparsified = Sparsifier::new(&g);
        assert_eq!(sparsified.next().unwrap().vertex_size(), N);
        assert_eq!(sparsified.next().unwrap().vertex_size(), N);
    }

    #[quickcheck]
    fn random_graph(g: RandomGraph) {
        let g = &g.graph;
        let mut size_v = 0;
        let spar = Sparsifier::new(g);
        spar.for_each(|vs| {
            size_v += vs.vertex_size();
        });
        assert_eq!(size_v, g.vertex_size());
    }

    #[derive(Debug, Clone)]
    struct RandomGraph {
        graph: undirected::TreeBackedGraph,
    }

    impl quickcheck::Arbitrary for RandomGraph {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            const N: usize = 10;

            let n: usize = usize::arbitrary(g) % N;
            let mut graph = undirected::TreeBackedGraph::new();
            let vertices: Vec<_> = (0..n).map(|_| graph.add_vertex()).collect();
            for _ in 0..(n.isqrt()) {
                let v0 = vertices[usize::arbitrary(g) % vertices.len()];
                let v1 = vertices[usize::arbitrary(g) % vertices.len()];
                graph.add_edge(v0, v1);
            }
            Self { graph }
        }
    }
}
