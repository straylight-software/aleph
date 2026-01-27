// src/dice-rs/bench/src/main.rs
//
// DICE benchmark - compare with Haskell DICE implementation
//
// Scenarios:
//   1. Linear dependency chain: a -> b -> c -> ... -> z
//   2. Diamond dependencies: many nodes depending on same base
//   3. Wide parallel: many independent computations
//   4. Incremental update: change one input, recompute

use std::time::{Duration, Instant};

use allocative::Allocative;
use async_trait::async_trait;
use derive_more::Display;
use dice::{DetectCycles, Dice, DiceComputations, InjectedKey, Key};
use dice_futures::cancellation::CancellationContext;
use dupe::Dupe;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Keys
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Injected base value
#[derive(Clone, Dupe, Display, Debug, Eq, Hash, PartialEq, Allocative)]
#[display("Base({})", _0)]
struct BaseKey(usize);

impl InjectedKey for BaseKey {
    type Value = i64;
    fn equality(x: &Self::Value, y: &Self::Value) -> bool {
        x == y
    }
}

/// Linear chain computation: depends on previous node
#[derive(Clone, Dupe, Display, Debug, Eq, Hash, PartialEq, Allocative)]
#[display("Chain({})", _0)]
struct ChainKey(usize);

#[async_trait]
impl Key for ChainKey {
    type Value = i64;

    async fn compute(
        &self,
        ctx: &mut DiceComputations,
        _cancellations: &CancellationContext,
    ) -> Self::Value {
        if self.0 == 0 {
            ctx.compute(&BaseKey(0)).await.unwrap()
        } else {
            let prev = ctx.compute(&ChainKey(self.0 - 1)).await.unwrap();
            prev + 1
        }
    }

    fn equality(x: &Self::Value, y: &Self::Value) -> bool {
        x == y
    }
}

/// Diamond dependency: depends on base
#[derive(Clone, Dupe, Display, Debug, Eq, Hash, PartialEq, Allocative)]
#[display("Diamond({})", _0)]
struct DiamondKey(usize);

#[async_trait]
impl Key for DiamondKey {
    type Value = i64;

    async fn compute(
        &self,
        ctx: &mut DiceComputations,
        _cancellations: &CancellationContext,
    ) -> Self::Value {
        let base = ctx.compute(&BaseKey(0)).await.unwrap();
        base * (self.0 as i64)
    }

    fn equality(x: &Self::Value, y: &Self::Value) -> bool {
        x == y
    }
}

/// Independent computation (no deps)
#[derive(Clone, Dupe, Display, Debug, Eq, Hash, PartialEq, Allocative)]
#[display("Independent({})", _0)]
struct IndependentKey(usize);

#[async_trait]
impl Key for IndependentKey {
    type Value = i64;

    async fn compute(
        &self,
        _ctx: &mut DiceComputations,
        _cancellations: &CancellationContext,
    ) -> Self::Value {
        // Simulate some work
        let mut sum: i64 = 0;
        for i in 0..1000 {
            sum = sum.wrapping_add(i * self.0 as i64);
        }
        sum
    }

    fn equality(x: &Self::Value, y: &Self::Value) -> bool {
        x == y
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Benchmarks
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct BenchResult {
    name: String,
    iterations: usize,
    total_time: Duration,
    per_iter: Duration,
}

impl BenchResult {
    fn print(&self) {
        println!(
            "{:30} {:>6} iters  {:>10.3?} total  {:>10.3?}/iter",
            self.name, self.iterations, self.total_time, self.per_iter
        );
    }
}

async fn bench_linear_chain(n: usize, iterations: usize) -> BenchResult {
    let dice = Dice::builder().build(DetectCycles::Disabled);
    
    // Setup
    let mut updater = dice.updater();
    updater.changed_to(vec![(BaseKey(0), 0i64)]).unwrap();
    let mut ctx = updater.commit().await;
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = ctx.compute(&ChainKey(n - 1)).await.unwrap();
    }
    let total = start.elapsed();
    
    BenchResult {
        name: format!("linear_chain({})", n),
        iterations,
        total_time: total,
        per_iter: total / iterations as u32,
    }
}

async fn bench_diamond(n: usize, iterations: usize) -> BenchResult {
    let dice = Dice::builder().build(DetectCycles::Disabled);
    
    // Setup
    let mut updater = dice.updater();
    updater.changed_to(vec![(BaseKey(0), 42i64)]).unwrap();
    let mut ctx = updater.commit().await;
    
    let start = Instant::now();
    for _ in 0..iterations {
        // Compute all diamond nodes
        for i in 0..n {
            let _result = ctx.compute(&DiamondKey(i)).await.unwrap();
        }
    }
    let total = start.elapsed();
    
    BenchResult {
        name: format!("diamond({})", n),
        iterations,
        total_time: total,
        per_iter: total / iterations as u32,
    }
}

async fn bench_parallel_independent(n: usize, iterations: usize) -> BenchResult {
    let dice = Dice::builder().build(DetectCycles::Disabled);
    
    let updater = dice.updater();
    let mut ctx = updater.commit().await;
    
    let start = Instant::now();
    for _ in 0..iterations {
        // Compute all independent nodes in parallel using compute_many
        let keys: Vec<_> = (0..n).map(IndependentKey).collect();
        let futs = ctx.compute_many(keys.iter().map(|k| {
            DiceComputations::declare_closure(
                move |ctx: &mut DiceComputations| -> futures::future::BoxFuture<i64> {
                    Box::pin(async move { ctx.compute(k).await.unwrap() })
                },
            )
        }));
        let _results: Vec<i64> = futures::future::join_all(futs).await;
    }
    let total = start.elapsed();
    
    BenchResult {
        name: format!("parallel_independent({})", n),
        iterations,
        total_time: total,
        per_iter: total / iterations as u32,
    }
}

async fn bench_incremental_update(chain_len: usize, iterations: usize) -> BenchResult {
    let dice = Dice::builder().build(DetectCycles::Disabled);
    
    // Initial setup
    let mut updater = dice.updater();
    updater.changed_to(vec![(BaseKey(0), 0i64)]).unwrap();
    let mut ctx = updater.commit().await;
    
    // First compute to populate cache
    let _result = ctx.compute(&ChainKey(chain_len - 1)).await.unwrap();
    
    let start = Instant::now();
    for i in 0..iterations {
        // Update base value
        let mut updater = dice.updater();
        updater.changed_to(vec![(BaseKey(0), i as i64)]).unwrap();
        ctx = updater.commit().await;
        
        // Recompute - should invalidate and recompute chain
        let _result = ctx.compute(&ChainKey(chain_len - 1)).await.unwrap();
    }
    let total = start.elapsed();
    
    BenchResult {
        name: format!("incremental_update(chain={})", chain_len),
        iterations,
        total_time: total,
        per_iter: total / iterations as u32,
    }
}

async fn bench_cache_hits(n: usize, iterations: usize) -> BenchResult {
    let dice = Dice::builder().build(DetectCycles::Disabled);
    
    // Setup
    let mut updater = dice.updater();
    updater.changed_to(vec![(BaseKey(0), 0i64)]).unwrap();
    let mut ctx = updater.commit().await;
    
    // First compute to populate cache
    let _result = ctx.compute(&ChainKey(n - 1)).await.unwrap();
    
    // Now measure cache hits
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = ctx.compute(&ChainKey(n - 1)).await.unwrap();
    }
    let total = start.elapsed();
    
    BenchResult {
        name: format!("cache_hits(chain={})", n),
        iterations,
        total_time: total,
        per_iter: total / iterations as u32,
    }
}

#[tokio::main]
async fn main() {
    println!("DICE Benchmark");
    println!("==============\n");
    
    // Warm up
    let _ = bench_linear_chain(10, 10).await;
    
    println!("Scenario 1: Linear dependency chain");
    println!("------------------------------------");
    bench_linear_chain(10, 1000).await.print();
    bench_linear_chain(100, 100).await.print();
    bench_linear_chain(1000, 10).await.print();
    println!();
    
    println!("Scenario 2: Diamond dependencies (many -> one)");
    println!("-----------------------------------------------");
    bench_diamond(10, 1000).await.print();
    bench_diamond(100, 100).await.print();
    bench_diamond(1000, 10).await.print();
    println!();
    
    println!("Scenario 3: Parallel independent computations");
    println!("----------------------------------------------");
    bench_parallel_independent(10, 1000).await.print();
    bench_parallel_independent(100, 100).await.print();
    bench_parallel_independent(1000, 10).await.print();
    println!();
    
    println!("Scenario 4: Incremental updates");
    println!("--------------------------------");
    bench_incremental_update(10, 100).await.print();
    bench_incremental_update(100, 100).await.print();
    bench_incremental_update(1000, 10).await.print();
    println!();
    
    println!("Scenario 5: Cache hits (no recomputation)");
    println!("------------------------------------------");
    bench_cache_hits(10, 10000).await.print();
    bench_cache_hits(100, 10000).await.print();
    bench_cache_hits(1000, 1000).await.print();
    println!();
}
