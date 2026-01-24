-- nix/modules/flake/build/scripts/buckconfig-python.dhall
--
-- Python toolchain configuration for Buck2
-- Environment variables are injected by render.dhall-with-vars

let interpreter : Text = env:INTERPRETER as Text
let python_include : Text = env:PYTHON_INCLUDE as Text
let python_lib : Text = env:PYTHON_LIB as Text
let nanobind_include : Text = env:NANOBIND_INCLUDE as Text
let nanobind_cmake : Text = env:NANOBIND_CMAKE as Text
let pybind11_include : Text = env:PYBIND11_INCLUDE as Text

in ''

[python]
interpreter = ${interpreter}
python_include = ${python_include}
python_lib = ${python_lib}
nanobind_include = ${nanobind_include}
nanobind_cmake = ${nanobind_cmake}
pybind11_include = ${pybind11_include}
''
