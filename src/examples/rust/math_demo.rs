// math_demo.rs
// Demonstrates using rust_library as a dependency
//
// Uses the mathlib crate to show dependency chaining in Buck2

extern crate mathlib;

fn main() {
    println!("Math Demo - Using mathlib crate");
    println!("================================");

    // Test basic operations
    let a = 10;
    let b = 5;
    println!("\nBasic operations:");
    println!("  add({}, {}) = {}", a, b, mathlib::add(a, b));
    println!("  multiply({}, {}) = {}", a, b, mathlib::multiply(a, b));

    // Test factorial
    println!("\nFactorials:");
    for n in 0..=10 {
        println!("  {}! = {}", n, mathlib::factorial(n));
    }

    // Test fibonacci
    println!("\nFibonacci sequence:");
    let fibs: Vec<u64> = (0..15).map(mathlib::fibonacci).collect();
    println!("  {:?}", fibs);

    println!("\nMath demo complete!");
}
