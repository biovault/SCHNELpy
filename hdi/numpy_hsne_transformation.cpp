#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"  // automatic conversion of STL to list, set, tuple, dict
#include "pybind11/stl_bind.h"
#include "cout_log.h"
#include "hierarchical_sne_inl.h"
#include "map_mem_eff.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

namespace py = pybind11;

bool numpy_to_hsne(
    py::array_t<float, py::array::c_style | py::array::forcecast> &X,
    const std::string &filePath,
    int num_scales,
    int seed,
    float landmark_threshold,
    int num_neighbors,
    int num_trees,
    int num_checks,
    float transition_matrix_prune_thresh,
    int num_walks,
    int num_walks_per_landmark,
    bool monte_carlo_sampling,
    bool out_of_core_computation
    ) {
    
    //hdi::utils::CoutLog log;
    hdi::dr::HierarchicalSNE<float, std::vector<hdi::data::MapMemEff<uint32_t,float>>>::Parameters params;
    std::vector< std::vector<float>* > _landmarkWeights;
    hdi::dr::HierarchicalSNE<float, std::vector<hdi::data::MapMemEff<uint32_t,float>>> _hsne;

    std::vector<hdi::data::MapMemEff<uint32_t, float>> *top_scale_matrix = nullptr;

    std::vector<uint64_t> placeholder (X.request().shape[0]);
    uint64_t *point_ids = placeholder.data();
    int num_point_ids = X.request().shape[0];

    params._seed = seed;
    params._mcmcs_landmark_thresh = landmark_threshold;
    params._num_neighbors = num_neighbors;
    params._aknn_num_trees = num_trees;
    params._aknn_num_checks = num_checks;
    params._transition_matrix_prune_thresh = transition_matrix_prune_thresh;
    params._mcmcs_num_walks = num_walks;
    params._num_walks_per_landmark = num_walks_per_landmark;

    params._monte_carlo_sampling = monte_carlo_sampling;
    params._out_of_core_computation = out_of_core_computation;

    if (nullptr == point_ids) {
        std::cout << "No point ids\n";
    }

    try {
        auto X_loc = X;
        py::buffer_info X_info = X_loc.request();
        if (X_info.ndim != 2) {
            throw std::runtime_error("Expecting input data to have two dimensions, data point and values");
        }
        int _num_data_points = X_info.shape[0];
        int _num_dimensions = X_info.shape[1];

        hdi::dr::HierarchicalSNE<float, std::vector<hdi::data::MapMemEff<uint32_t,float>>> _hsne;
        //_hsne.setLogger(&log);
        _hsne.setDimensionality(_num_dimensions);

        if (top_scale_matrix == nullptr) {
            _hsne.initialize(static_cast<float *>(X_info.ptr), _num_data_points, params);
            //std::cout << _hsne.hierarchy().size() << std::endl;
            //_hsne.statistics().log(&log);

            _landmarkWeights.resize(num_scales);

            for (int s = 0; s < num_scales; ++s) {
                _landmarkWeights[s] = NULL;
            }

            for (int s = 0; s < num_scales - 1; ++s) {
                _hsne.addScale();
            }
            
            hdi::utils::AbstractLog* logger;
            std::ofstream filebin (filePath, std::ios::binary); // binary format
            hdi::dr::IO::saveHSNE(_hsne, filebin, logger);
        } else {
            _hsne.initialize(*top_scale_matrix, params);
        }
    }
    catch (const std::exception& e) {
        std::cout << "Fatal error: " << e.what() << std::endl;
        return false;
    }

    return true;
}

PYBIND11_MODULE(numpy_to_hsne, m) {
    m.def("run", &numpy_to_hsne, "function which converts numpy array to HSNE hierarchy in form of .hsne file",
    py::arg("X"), py::arg("filepath"), py::arg("num_scales"), py::arg("seeds"), py::arg("landmark_threshold"), 
    py::arg("num_neighbors"), py::arg("num_trees"), py::arg("num_checks"), py::arg("trans_matrix_prune_threshold"),
    py::arg("num_walks"), py::arg("num_walks_per_landmark"), py::arg("monte_carlo_sampling"), 
    py::arg("out_of_core_computation"));
}