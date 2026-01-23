// example.cpp
// nanobind example: C++ functions callable from Python
//
// This demonstrates the pattern for Python/C++ interop:
// - Write performance-critical code in C++
// - Expose via nanobind bindings
// - Call from Python with zero-copy where possible

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace nb = nanobind;

// =============================================================================
// Pure C++ functions (the actual implementation)
// =============================================================================

namespace aleph {

// Simple function: add two numbers
int add(int a, int b) {
    return a + b;
}

// String manipulation
std::string greet(const std::string& name) {
    return "Hello from aleph, " + name + "!";
}

// Vector operations (demonstrates zero-copy potential)
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector sizes must match");
    }
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// Compute norm
double norm(const std::vector<double>& v) {
    return std::sqrt(dot_product(v, v));
}

// A simple class
class Counter {
public:
    Counter(int initial = 0) : value_(initial) {}

    void increment() { ++value_; }
    void decrement() { --value_; }
    void add(int n) { value_ += n; }
    int get() const { return value_; }
    void reset() { value_ = 0; }

    std::string repr() const {
        return "Counter(" + std::to_string(value_) + ")";
    }

private:
    int value_;
};

}  // namespace aleph

// =============================================================================
// nanobind module definition
// =============================================================================

NB_MODULE(example, m) {
    m.doc() = "aleph example module: C++ called from Python via nanobind";

    // Simple functions
    m.def("add", &aleph::add, "Add two integers",
          nb::arg("a"), nb::arg("b"));

    m.def("greet", &aleph::greet, "Generate a greeting",
          nb::arg("name"));

    // Vector functions
    m.def("dot_product", &aleph::dot_product,
          "Compute dot product of two vectors",
          nb::arg("a"), nb::arg("b"));

    m.def("norm", &aleph::norm,
          "Compute Euclidean norm of a vector",
          nb::arg("v"));

    // Class binding
    nb::class_<aleph::Counter>(m, "Counter")
        .def(nb::init<int>(), nb::arg("initial") = 0)
        .def("increment", &aleph::Counter::increment)
        .def("decrement", &aleph::Counter::decrement)
        .def("add", &aleph::Counter::add, nb::arg("n"))
        .def("get", &aleph::Counter::get)
        .def("reset", &aleph::Counter::reset)
        .def("__repr__", &aleph::Counter::repr);

    // Module-level test function (called by buck2 run)
    m.def("test", []() {
        // Run some basic tests
        bool ok = true;

        // Test add
        if (aleph::add(2, 3) != 5) {
            ok = false;
        }

        // Test dot product
        std::vector<double> a = {1.0, 2.0, 3.0};
        std::vector<double> b = {4.0, 5.0, 6.0};
        double expected = 32.0;  // 1*4 + 2*5 + 3*6
        if (std::abs(aleph::dot_product(a, b) - expected) > 1e-10) {
            ok = false;
        }

        // Test Counter
        aleph::Counter c(10);
        c.increment();
        if (c.get() != 11) {
            ok = false;
        }

        if (ok) {
            nb::print("example module: all tests passed");
        } else {
            nb::print("example module: TESTS FAILED");
        }
    });
}
