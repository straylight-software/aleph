/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is dual-licensed under either the MIT license found in the
 * LICENSE-MIT file in the root directory of this source tree or the Apache
 * License, Version 2.0 found in the LICENSE-APACHE file in the root directory
 * of this source tree. You may select, at your option, one of the
 * above-listed licenses.
 */

//! Manual implementations of `Allocative` for various types.
//!
//! NOTE: Module names suffixed with `_impls` to avoid shadowing extern crates
//! of the same name. This is necessary because Rust resolves local module
//! names before extern crate names.

mod anyhow_impls;
mod bumpalo;
pub(crate) mod common;
mod compact_str;
mod dashmap_impls;
mod either;
mod futures_impls;
pub(crate) mod hashbrown;
pub(crate) mod hashbrown_util;
mod indexmap;
mod lock_api_impls;
mod num_bigint;
mod once_cell_impls;
mod parking_lot_impls;
mod prost_types;
mod relative_path;
mod serde_json;
mod slab_impls;
mod smallvec;
mod sorted_vector_map;
mod std;
mod tokio_impls;
mod triomphe_impls;
