-- Drv/Flags.dhall
-- Sum types for build configuration. No strings.

let BuildType = < Release | Debug | RelWithDebInfo | MinSizeRel >

let Linkage = < Static | Shared | Both >

let Optimization = < O0 | O1 | O2 | O3 | Os | Oz >

let Sanitizer = < Address | Thread | Memory | UndefinedBehavior | None >

let LTO = < Off | Thin | Full >

let PIC = < On | Off | Default >

let SIMD = < None | SSE2 | SSE4 | AVX | AVX2 | AVX512 | Neon >

in { BuildType, Linkage, Optimization, Sanitizer, LTO, PIC, SIMD }
