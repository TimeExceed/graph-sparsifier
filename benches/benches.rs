use algograph::graph::*;
use criterion::*;
use graph_sparsifier::page_rank::approx::Sparsifier;
use rand::{prelude::*, rngs::SmallRng};

criterion_main!(benches);
criterion_group!(
    benches,
    clique,
    rope,
    rope_head,
    random_graph,
    random_graph_head,
    grape,
    grape_head
);

fn clique(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clique");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);
    const SIZES: &[usize] = &[10usize, 20usize, 40usize, 80usize, 160usize, 320usize];
    for n in SIZES.iter() {
        let mut g = undirected::TreeBackedGraph::new();
        let _ = add_clique(&mut g, *n);
        group.bench_with_input(BenchmarkId::new("ApproxPR", n), n, |b, _| {
            b.iter(|| {
                let spar = Sparsifier::new(&g);
                spar.for_each(|_| {});
            })
        });
    }
    group.finish();
}

fn rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rope");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);
    const SIZES: &[usize] = &[10usize, 20usize, 40usize, 80usize, 160usize];
    for n in SIZES.iter() {
        let mut g = undirected::TreeBackedGraph::new();
        let mut u = g.add_vertex();
        for _ in 0..*n {
            let v = g.add_vertex();
            g.add_edge(u, v);
            u = v;
        }
        group.bench_with_input(BenchmarkId::new("ApproxPR", n), n, |b, _| {
            b.iter(|| {
                let spar = Sparsifier::new(&g);
                spar.for_each(|_| {});
            })
        });
    }
    group.finish();
}

fn rope_head(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("RopeHead");
    group.plot_config(plot_config);
    const SIZES: &[usize] = &[10usize, 20usize, 40usize, 80usize, 160usize, 320usize];
    for n in SIZES.iter() {
        let mut g = undirected::TreeBackedGraph::new();
        let mut u = g.add_vertex();
        for _ in 0..*n {
            let v = g.add_vertex();
            g.add_edge(u, v);
            u = v;
        }
        group.bench_with_input(BenchmarkId::new("ApproxPR", n), n, |b, _| {
            b.iter(|| {
                let mut spar = Sparsifier::new(&g);
                black_box(spar.next());
            })
        });
    }
    group.finish();
}

fn random_graph(c: &mut Criterion) {
    const V_SIZE: &[usize] = &[10usize, 20usize, 40usize, 80usize];
    const E_POW: &[f64] = &[1.0, 1.25, 1.5];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut rng = SmallRng::seed_from_u64(3407);
    for e_m in E_POW.iter() {
        let mut group = c.benchmark_group(format!("RandomGraph_{e_m:.2}"));
        group.plot_config(plot_config.clone());
        for v_n in V_SIZE.iter() {
            let e_n = (*v_n as f64).powf(*e_m) as usize;
            let g = gen_random_graph(&mut rng, *v_n, e_n);
            group.bench_with_input(BenchmarkId::new("ApproxPR", v_n), v_n, |b, _| {
                b.iter(|| {
                    let spar = Sparsifier::new(&g);
                    spar.for_each(|_| {});
                })
            });
        }
        group.finish();
    }
}

fn random_graph_head(c: &mut Criterion) {
    const V_SIZE: &[usize] = &[10usize, 20usize, 40usize, 80usize, 160usize];
    const E_POW: &[f64] = &[1.0, 1.25, 1.5];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut rng = SmallRng::seed_from_u64(3407);
    for e_m in E_POW.iter() {
        let mut group = c.benchmark_group(format!("RandomGraphHead_{e_m:.2}"));
        group.plot_config(plot_config.clone());
        for v_n in V_SIZE.iter() {
            let e_n = (*v_n as f64).powf(*e_m) as usize;
            let g = gen_random_graph(&mut rng, *v_n, e_n);
            group.bench_with_input(BenchmarkId::new("ApproxPR", v_n), v_n, |b, _| {
                b.iter(|| {
                    let mut spar = Sparsifier::new(&g);
                    black_box(spar.next());
                })
            });
        }
        group.finish();
    }
}

fn grape(c: &mut Criterion) {
    const N: usize = 10;
    const M: &[usize] = &[10usize, 20usize, 40usize, 80usize];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Grape".to_string());
    group.plot_config(plot_config);
    for m in M.iter() {
        let mut g = undirected::TreeBackedGraph::new();
        let mut vs = add_clique(&mut g, N);
        for _ in 1..*m {
            let tail = vs.pop().unwrap();
            vs = add_clique(&mut g, N);
            let head = *vs.first().unwrap();
            g.add_edge(tail, head);
        }
        group.bench_with_input(BenchmarkId::new("ApproxPR", m), m, |b, _| {
            b.iter(|| {
                let spar = Sparsifier::new(&g);
                spar.for_each(|_| {});
            })
        });
    }
    group.finish();
}

fn grape_head(c: &mut Criterion) {
    const N: usize = 10;
    const M: &[usize] = &[10usize, 20usize, 40usize, 80usize];
    let mut group = c.benchmark_group("GrapeHead".to_string());
    for m in M.iter() {
        let mut g = undirected::TreeBackedGraph::new();
        let mut vs = add_clique(&mut g, N);
        for _ in 1..*m {
            let tail = vs.pop().unwrap();
            vs = add_clique(&mut g, N);
            let head = *vs.first().unwrap();
            g.add_edge(tail, head);
        }
        group.bench_with_input(BenchmarkId::new("ApproxPR", m), m, |b, _| {
            b.iter(|| {
                let mut spar = Sparsifier::new(&g);
                black_box(spar.next());
            })
        });
    }
    group.finish();
}

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn add_clique<G: GrowableGraph>(g: &mut G, n: usize) -> Vec<VertexId> {
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

fn gen_random_graph<R>(rng: &mut R, v_n: usize, e_n: usize) -> undirected::TreeBackedGraph
where
    R: SeedableRng + Rng,
{
    let mut g = undirected::TreeBackedGraph::new();
    let vs: Vec<_> = (0..v_n).map(|_| g.add_vertex()).collect();
    for _ in 0..e_n {
        let u = *vs.choose(rng).unwrap();
        let v = *vs.choose(rng).unwrap();
        g.add_edge(u, v);
    }
    g
}
