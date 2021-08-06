#pragma once

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#include <chrono>
#include <iostream>
#include <iterator>

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

inline std::pair<std::vector<int32_t>, std::vector<int32_t>> cpu_FindNeighborEdges(
  int const iSearchDistance,
  std::vector<float> const &v_x,
  std::vector<float> const &v_y,
  std::vector<float> const &v_x_1,
  std::vector<float> const &v_y_1) {
  std::cout << " Number of edges " << v_x.size() << std::endl;
  if (v_x.empty()) { return std::make_pair(std::vector<int32_t>{}, std::vector<int32_t>{}); }

  typedef bg::model::point<int, 2, bg::cs::cartesian> point;
  typedef bg::model::box<point> box;
  typedef std::pair<box, unsigned> value;

  // create the rtree using default constructor
  bgi::rtree<value, bgi::quadratic<16>> rtree;

  auto startTimeQTreeConstruction = std::chrono::high_resolution_clock::now();
  std::vector<value> vPairs;
  std::vector<box> vBoxes;
  // create some values
  for (unsigned i = 0; i < v_x.size(); ++i) {
    auto min_x = std::min(v_x[i], v_x_1[i]);
    auto max_x = std::max(v_x[i], v_x_1[i]);
    auto min_y = std::min(v_y[i], v_y_1[i]);
    auto max_y = std::max(v_y[i], v_y_1[i]);
    // create a box
    box b(point(min_x, min_y), point(max_x, max_y));
    vBoxes.emplace_back(b);
    // insert new value
    rtree.insert(std::make_pair(b, i));
    vPairs.emplace_back(std::make_pair(b, i));
  }
  auto finishTimeQTreeConstruction = std::chrono::high_resolution_clock::now();
  std::cout << " Search CPU Rtree quadratic construction time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(finishTimeQTreeConstruction -
                                                                     startTimeQTreeConstruction)
                   .count() /
                 1000
            << " ms\n";

  startTimeQTreeConstruction = std::chrono::high_resolution_clock::now();
  bgi::rtree<value, bgi::quadratic<16>> rtree_bulk(vPairs);
  finishTimeQTreeConstruction = std::chrono::high_resolution_clock::now();
  std::cout << " Search CPU RTree bulk loading construction time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(finishTimeQTreeConstruction -
                                                                     startTimeQTreeConstruction)
                   .count() /
                 1000
            << " ms\n";

  auto data_box_ids  = std::vector<int32_t>{};
  auto query_box_ids = std::vector<int32_t>{};
  // find values intersecting some area defined by a box
  double queryTime = 0.;
  for (unsigned i = 0; i < vBoxes.size(); i++) {
    auto box  = vBoxes[i];
    int min_x = box.min_corner().get<0>() - iSearchDistance;
    int min_y = box.min_corner().get<1>() - iSearchDistance;
    int max_x = box.max_corner().get<0>() + iSearchDistance;
    int max_y = box.max_corner().get<1>() + iSearchDistance;

    startTimeQTreeConstruction = std::chrono::high_resolution_clock::now();
    bg::model::box<point> query_box;
    query_box.min_corner().set<0>(min_x);
    query_box.min_corner().set<1>(min_y);
    query_box.max_corner().set<0>(max_x);
    query_box.max_corner().set<1>(max_y);
    std::vector<value> result_s;
    rtree.query(bgi::intersects(query_box), std::back_inserter(result_s));
    finishTimeQTreeConstruction = std::chrono::high_resolution_clock::now();
    queryTime += std::chrono::duration_cast<std::chrono::microseconds>(finishTimeQTreeConstruction -
                                                                       startTimeQTreeConstruction)
                   .count() /
                 1000.;

    std::vector<int32_t> result_ids;
    std::transform(
      result_s.begin(), result_s.end(), std::back_inserter(result_ids), [](auto const &x) {
        return x.second;
      });
    std::sort(result_ids.begin(), result_ids.end());

    for (auto id : result_ids) {
      data_box_ids.push_back(id);
      query_box_ids.push_back(i);
    }
  }
  std::cout << " Search CPU RTree query time: " << queryTime << " ms\n";

  // display results
  /*
  std::cout << "spatial query box:" << std::endl;
  std::cout << bg::wkt<box>(query_box) << std::endl;
  std::cout << "spatial query result:" << std::endl;
  BOOST_FOREACH(value const& v, result_s)
      std::cout << bg::wkt<box>(v.first) << " - " << v.second << std::endl;

  std::cout << "knn query point:" << std::endl;
  std::cout << bg::wkt<point>(point(0, 0)) << std::endl;
  std::cout << "knn query result:" << std::endl;
  BOOST_FOREACH(value const& v, result_n)
      std::cout << bg::wkt<box>(v.first) << " - " << v.second << std::endl;
      */

  return std::make_pair(std::move(data_box_ids), std::move(query_box_ids));
}
