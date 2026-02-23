// pybind libraries
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// C/C++ includes
#include <cfloat>
#include <chrono>
#include <cmath>
#include <vector>

namespace py = pybind11;
using namespace std;

Eigen::MatrixXf rgb_map(
    const Eigen::MatrixXf& points,     /* Nx4 ndarray */
    const Eigen::VectorXf lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
    double bev_res /* m per pixel*/)
{
    auto xmin = lidar_range[0];
    auto xmax = lidar_range[1];
    auto ymin = lidar_range[2];
    auto ymax = lidar_range[3];
    auto zmin = lidar_range[4];
    auto zmax = lidar_range[5];

    int img_w = int((xmax - xmin) / bev_res);
    int img_h = int((ymax - ymin) / bev_res);
    Eigen::MatrixXf bev_map = Eigen::MatrixXf::Zero(img_h * img_w, 3);

    for (ssize_t n = 0; n < points.rows(); n++) {
        auto x = points(n, 0);
        if (x < xmin || x > xmax)
            continue;
        auto y = points(n, 1);
        if (y < ymin || y > ymax)
            continue;
        auto z = points(n, 2);
        if (z < zmin || z > zmax)
            continue;

        auto i = points(n, 3);
        // convert to ego-car coordinate
        int px = floor((x - xmin) / bev_res);
        int py = floor((ymax - y) / bev_res);
        double h = (z - zmin) / (zmax - zmin);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);

        int idx = py * img_w + px;
        bev_map(idx, 0) += i;
        if (h > bev_map(idx, 1))
            bev_map(idx, 1) = h;
        bev_map(idx, 2) += 1;
    }

    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            int idx = py * img_w + px;
            auto c = bev_map(idx, 2);
            if (c > 0) {
                bev_map(idx, 0) /= c;
            }
            bev_map(idx, 2) = std::min(1.0, log(c + 1) / log(64));
        }
    }
    return bev_map;
}

Eigen::MatrixXf rgb_label_map(
    const Eigen::MatrixXf& points,     /* Nx4 ndarray */
    const Eigen::MatrixXi& labels,     /* Nx1 ndarray */
    const Eigen::VectorXf lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
    double bev_res /* m per pixel*/)
{
    auto xmin = lidar_range[0];
    auto xmax = lidar_range[1];
    auto ymin = lidar_range[2];
    auto ymax = lidar_range[3];
    auto zmin = lidar_range[4];
    auto zmax = lidar_range[5];

    int img_w = int((xmax - xmin) / bev_res);
    int img_h = int((ymax - ymin) / bev_res);
    Eigen::MatrixXf bev_map = Eigen::MatrixXf::Zero(img_h * img_w, 4);

    for (ssize_t n = 0; n < points.rows(); n++) {
        auto x = points(n, 0);
        if (x < xmin || x > xmax)
            continue;
        auto y = points(n, 1);
        if (y < ymin || y > ymax)
            continue;
        auto z = points(n, 2);
        if (z < zmin || z > zmax)
            continue;

        auto l = labels(n, 0);
        auto i = points(n, 3);
        // convert to ego-car coordinate
        int px = floor((x - xmin) / bev_res);
        int py = floor((ymax - y) / bev_res);
        double h = (z - zmin) / (zmax - zmin);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);

        int idx = py * img_w + px;
        bev_map(idx, 0) += i;
        if (h > bev_map(idx, 1)) {
            bev_map(idx, 1) = h;
            bev_map(idx, 3) = l;
        }
        bev_map(idx, 2) += 1;
    }

    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            int idx = py * img_w + px;
            auto c = bev_map(idx, 2);
            if (c > 0) {
                bev_map(idx, 0) /= c;
            }
            bev_map(idx, 2) = std::min(1.0, log(c + 1) / log(64));
        }
    }
    return bev_map;
}

Eigen::MatrixXf rgb_label_map_3d(
    const Eigen::MatrixXf& points,     /* Nx4 ndarray */
    const Eigen::MatrixXi& labels,     /* Nx1 ndarray */
    const Eigen::VectorXf lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
    double bev_res /* m per pixel*/,
    double bev_res_z /* m per pixel*/)
{
    auto xmin = lidar_range[0];
    auto xmax = lidar_range[1];
    auto ymin = lidar_range[2];
    auto ymax = lidar_range[3];
    auto zmin = lidar_range[4];
    auto zmax = lidar_range[5];

    int img_w = int((xmax - xmin) / bev_res);
    int img_h = int((ymax - ymin) / bev_res);
    int img_c = int((zmax - zmin) / bev_res_z);
    Eigen::MatrixXf bev_map = Eigen::MatrixXf::Zero(img_h * img_w * img_c, 4);

    for (ssize_t n = 0; n < points.rows(); n++) {
        auto x = points(n, 0);
        if (x < xmin || x > xmax)
            continue;
        auto y = points(n, 1);
        if (y < ymin || y > ymax)
            continue;
        auto z = points(n, 2);
        if (z < zmin || z > zmax)
            continue;

        // convert to ego-car coordinate
        int px = floor((x - xmin) / bev_res);
        int py = floor((ymax - y) / bev_res);
        int pz = floor((z - zmin) / bev_res_z);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);
        pz = std::min(pz, img_c - 1);
        int idx = py * img_w * img_c + px * img_c + pz;

        auto i = points(n, 3);
        auto l = labels(n, 0);
        double h = (z - (pz * bev_res_z + zmin)) / bev_res_z;
        bev_map(idx, 0) += i;
        bev_map(idx, 2) += 1;
        if (h > bev_map(idx, 1)) {
            bev_map(idx, 1) = h;
            bev_map(idx, 3) = l;
        }
    }

    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            for (ssize_t pz = 0; pz < img_c; pz++) {
                int idx = py * img_w * img_c + px * img_c + pz;
                auto c = bev_map(idx, 2);
                if (c > 0) {
                    bev_map(idx, 0) /= c;
                }
                bev_map(idx, 2) = std::min(1.0, log(c + 1) / log(64));
            }
        }
    }
    return bev_map;
}

PYBIND11_MODULE(bev, m)
{
    m.doc() = "Create BEV map from lidar points";

    m.def("rgb_map",
        &rgb_map,
        py::arg("points"),
        py::arg("lidar_range"),
        py::arg("bev_res"));

    m.def("rgb_label_map",
        &rgb_label_map,
        py::arg("points"),
        py::arg("labels"),
        py::arg("lidar_range"),
        py::arg("bev_res"));

    m.def("rgb_label_map_3d",
        &rgb_label_map_3d,
        py::arg("points"),
        py::arg("labels"),
        py::arg("lidar_range"),
        py::arg("bev_res"),
        py::arg("bev_res_z"));
}
