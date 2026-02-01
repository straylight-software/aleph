// Test using once_cell from crates.io
use once_cell::sync::Lazy;

static GREETING: Lazy<String> = Lazy::new(|| {
    "Hello from crates.io!".to_string()
});

fn main() {
    println!("{}", *GREETING);
}
