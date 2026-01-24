#pragma once
#include <experimental/mdspan>

namespace std {
using experimental::mdspan;
using experimental::extents;
using experimental::dextents;
using experimental::layout_right;
using experimental::layout_left;
using experimental::layout_stride;
using experimental::default_accessor;
using experimental::full_extent;
using experimental::submdspan;
}
