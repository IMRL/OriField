// pybind libraries
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// C/C++ includes
#include <Eigen/Dense>
#include <iostream>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <vector>
#include <queue>
#include <stack>
#include <tuple>
#include <limits>
#include <random>
#include <algorithm>
#include <ctime>

#include <dubins.h>
// #include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace std;

struct Pixel {
    int x, y;
    Pixel(int x, int y) : x(x), y(y) {}
};

bool operator>(const Pixel& a, const Pixel& b) {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}

bool operator<(const Pixel& a, const Pixel& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

typedef std::pair<float, Pixel> type_pq_item; 

Eigen::MatrixXi dijkstra(
    const Eigen::MatrixXi& traversability,     /* HxW ndarray */
    const Eigen::MatrixXi& starts, /* Lx2 ndarray */
    const Eigen::MatrixXi& directions, /* 8x2 ndarray */
    const Eigen::VectorXf& costs /* 8 ndarray */
    )
{
    auto img_h = traversability.rows();
    auto img_w = traversability.cols();
    Eigen::MatrixXi visit(traversability);
    Eigen::MatrixXf distances = Eigen::MatrixXf::Ones(img_h, img_w) * std::numeric_limits<int>::max();
    Eigen::MatrixXi movements = Eigen::MatrixXi::Zero(img_h, img_w);
    
    std::priority_queue<type_pq_item, std::vector<type_pq_item>, std::greater<type_pq_item>> queue;
    for (ssize_t n = 0; n < starts.rows(); n++) {
        distances(starts(n, 0), starts(n, 1)) = 0;
        queue.push({0, Pixel(starts(n, 0), starts(n, 1))});
    }

    while (!queue.empty()) {
        auto current = queue.top();
        queue.pop();
        
        auto current_distance = current.first;
        auto current_pixel = current.second;
        visit(current_pixel.x, current_pixel.y) = 2;  // visited

        for (ssize_t d = 0; d < directions.rows(); d++) {
            Pixel neighbor(current_pixel.x + directions(d, 0), current_pixel.y + directions(d, 1));
            if ((0 <= neighbor.x) && (neighbor.x < img_h) && (0 <= neighbor.y) && (neighbor.y < img_w)) {
                if (visit(neighbor.x, neighbor.y) == 1) {  // to visit
                    auto tentative_distance = current_distance + costs(d);
                    if (tentative_distance < distances(neighbor.x, neighbor.y)) {
                        distances(neighbor.x, neighbor.y) = tentative_distance;
                        queue.push({tentative_distance, neighbor});
                        movements(neighbor.x, neighbor.y) = d+1;
                    }
                }
            }  // Check bounds
        }
    }
    
    return movements;
}

Eigen::MatrixXi randomwalk(
    const Eigen::MatrixXi& traversability,     /* HxW ndarray */
    const Eigen::MatrixXi& starts, /* Lx2 ndarray */
    const Eigen::MatrixXi& directions, /* 8x2 ndarray */
    const Eigen::VectorXf& costs /* 8 ndarray */
    )
{
    auto img_h = traversability.rows();
    auto img_w = traversability.cols();
    Eigen::MatrixXi visit(traversability);
    Eigen::MatrixXf distances = Eigen::MatrixXf::Ones(img_h, img_w) * std::numeric_limits<int>::max();
    Eigen::MatrixXi movements = Eigen::MatrixXi::Zero(img_h, img_w);
    
    std::stack<Pixel> stack;
    for (ssize_t n = 0; n < starts.rows(); n++) {
        distances(starts(n, 0), starts(n, 1)) = 0;
        stack.push(Pixel(starts(n, 0), starts(n, 1)));
    }

    while (!stack.empty()) {
        Pixel current_pixel = stack.top();
        auto current_distance = distances(current_pixel.x, current_pixel.y);
        visit(current_pixel.x, current_pixel.y) = 2;  // visited

        std::vector<int> ds = {0, 1, 2, 3, 4, 5, 6, 7};
        std::random_shuffle(ds.begin(), ds.end()); // Randomize the order of directions
        
        bool moved = false;
        for (int d : ds) {
            Pixel neighbor(current_pixel.x + directions(d, 0), current_pixel.y + directions(d, 1));
            if ((0 <= neighbor.x) && (neighbor.x < img_h) && (0 <= neighbor.y) && (neighbor.y < img_w)) {
                if (visit(neighbor.x, neighbor.y) == 1) {  // to visit
                    distances(neighbor.x, neighbor.y) = current_distance + costs(d);
                    stack.push(neighbor);
                    movements(neighbor.x, neighbor.y) = d+1;
                    moved = true;
                    break;
                }
            }
        }

        if (!moved) {
            stack.pop(); // Pop from stack if no valid moves
        }
    }
    
    return movements;
}

struct Grid {
    int x, y;
    double distance(const Grid& b) const {
        return std::sqrt(std::pow(x - b.x, 2) + std::pow(y - b.y, 2));
    }
    double meter(const Grid& b, double res) const {
        return std::sqrt(std::pow(x*res - b.x*res, 2) + std::pow(y*res - b.y*res, 2));
    }
    bool operator==(const Grid& b) const {
        return (x == b.x) && (y == b.y);
    }
};

std::ostream& operator<<(std::ostream& os, const Grid& g)
{
    os << "x: " << g.x << ", y: " << g.y;
    return os;
}

struct Point {
    double x, y;
    double distance(const Point& b) const {
        return std::sqrt(std::pow(x - b.x, 2) + std::pow(y - b.y, 2));
    }
    Grid toGrid(int img_h, int img_w) const {
        return {int(x * img_h), int(y * img_w)};
    }
};

std::ostream& operator<<(std::ostream& os, const Point& p)
{
    os << "x: " << p.x << ", y: " << p.y;
    return os;
}

struct Tangent {
    double x, y;
    Tangent operator+(const Tangent& b) const {
        return {x + b.x, y + b.y};
    }
    Tangent operator-(const Tangent& b) const {
        Tangent a = {x, y};
        Tangent minus_b = {-b.x, -b.y};
        return a + minus_b;
    }
    Tangent operator*(double scale) const {
        return {x * scale, y * scale};
    }
    Tangent operator/(double scale) const {
        Tangent a = {x, y};
        double invert_scale = 1 / scale;
        return a * invert_scale;
    }
    double dot(const Tangent& b) const {
        return x*b.x + y*b.y;
    }
    double len() const {
        return std::sqrt(std::pow(x, 2) + std::pow(y, 2));
    }
    Tangent rotate(double radian) const {
        double cosRadian = std::cos(radian);
        double sinRadian = std::sin(radian);
        double newX = x * cosRadian - y * sinRadian;
        double newY = x * sinRadian + y * cosRadian;
        return {newX, newY};
    }
    Tangent norm() const {
        double l = len();
        if (l == 0) {
            return {0, 0};
        } else {
            return {x / l, y / l};
        }
    }
};

std::ostream& operator<<(std::ostream& os, const Tangent& t)
{
    os << "x: " << t.x << ", y: " << t.y;
    return os;
}

struct DataItem {
    double cost;
    double energy;
    int step;
    DataItem operator+(const DataItem& b) const {
        return {cost + b.cost, energy + b.energy, step + b.step};
    }
    bool operator==(const DataItem& b) const {
        return (cost == b.cost) && (energy == b.energy) && (step == b.step);
    }
    friend std::ostream& operator<<(std::ostream& os, const DataItem& dt);
};

std::ostream& operator<<(std::ostream& os, const DataItem& dt)
{
    os << "cost: " << dt.cost << ", energy: " << dt.energy << ", step: " << dt.step;
    return os;
}

void getMapConf(const std::vector<Grid>& lineGrids, const Eigen::MatrixXf& conf, std::vector<float>& res) {
    for (size_t i = 0; i < lineGrids.size(); i++) {
        float grid_conf = conf(lineGrids[i].x, lineGrids[i].y);
        res.push_back(grid_conf);
    }
}

void getMapTengent(const std::vector<Grid>& lineGrids, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, float rotation_factor, std::vector<Tangent>& res) {
    for (size_t i = 0; i < lineGrids.size(); i++) {
        double grid_tangent_x = tangent_x(lineGrids[i].x, lineGrids[i].y);
        double grid_tangent_y = tangent_y(lineGrids[i].x, lineGrids[i].y);
        Tangent grid_tangent({grid_tangent_x, grid_tangent_y});
        res.push_back(grid_tangent.rotate(rotation_factor));
    }
}

void getMapDiscorage(const std::vector<Grid>& lineGrids, const Eigen::MatrixXf& discorage, std::vector<float>& res) {
    for (size_t i = 0; i < lineGrids.size(); i++) {
        float grid_discorage = discorage(lineGrids[i].x, lineGrids[i].y);
        res.push_back(grid_discorage);
    }
}

std::vector<Grid> drawLineGrids(Grid A, Grid B) {
    std::vector<Grid> lineGrids_;
    int dx = std::abs(B.x - A.x), sx = A.x < B.x ? 1 : -1;
    int dy = -std::abs(B.y - A.y), sy = A.y < B.y ? 1 : -1;
    int err = dx + dy, e2; // error value e_xy

    while (true) {
        lineGrids_.push_back(A); // include the current point

        if (A.x == B.x && A.y == B.y) break; // the line has reached point B

        e2 = 2 * err;
        if (e2 >= dy) { // e_xy+e_x > 0
            err += dy;
            A.x += sx;
        }
        if (e2 <= dx) { // e_xy+e_y < 0
            err += dx;
            A.y += sy;
        }
    }
    return lineGrids_;
}

Tangent drawLineTengent(const Grid& startGrid, const Grid& endGrid) {
    double line_tangent_length = std::sqrt(std::pow(endGrid.x - startGrid.x, 2) + std::pow(endGrid.y - startGrid.y, 2));
    double line_tangent_x = (endGrid.x - startGrid.x) / line_tangent_length;
    double line_tangent_y = (endGrid.y - startGrid.y) / line_tangent_length;
    return {line_tangent_x, line_tangent_y};
}

// Callback function for sampling
int dubins_callback(double q[3], double t, void* user_data) {
    // printf("%f, %f, %f, %f\n", q[0], q[1], q[2], t);

    auto* data = static_cast<std::pair<std::vector<Eigen::Vector2f>*, std::vector<Eigen::Vector2f>*>*>(user_data);
    auto& points = data->first;
    auto& tangents = data->second;

    float x = static_cast<float>(q[0]);
    float y = static_cast<float>(q[1]);
    Eigen::Vector2f point(x, y);
    points->emplace_back(point);

    float theta = static_cast<float>(q[2]);
    Eigen::Vector2f tangent(std::cos(theta), std::sin(theta));
    tangents->emplace_back(tangent);

    return 0;
}

void drawDubinsPath(double start[3], double end[3], double turning_radius, std::vector<Eigen::Vector2f>& points, std::vector<Eigen::Vector2f>& tangents) {
    // Dubins path
    DubinsPath path;
    if (dubins_shortest_path(&path, start, end, turning_radius) != 0) {
        std::cout << "Error calculating Dubins path" << std::endl;
        return;
    }

    // Sample the path
    std::pair<std::vector<Eigen::Vector2f>*, std::vector<Eigen::Vector2f>*> data(&points, &tangents);
    dubins_path_sample_many(&path, 0.5, dubins_callback, &data);

    // // Create an image
    // int img_width = 120;
    // int img_height = 120;
    // cv::Mat image = cv::Mat::zeros(img_height, img_width, CV_8UC3);

    // // Draw the path on the image
    // for (size_t i = 1; i < points.size(); ++i) {
    //     cv::line(image, cv::Point(points[i - 1].x(), points[i - 1].y()),
    //              cv::Point(points[i].x(), points[i].y()), cv::Scalar(255, 0, 0), 1);
    // }

    // // Optionally, draw tangents (as small lines) for visualization
    // for (size_t i = 0; i < points.size(); ++i) {
    //     cv::Point start_point(points[i].x(), points[i].y());
    //     cv::Point end_point(start_point.x + tangents[i].x() * 5, start_point.y + tangents[i].y() * 5);
    //     cv::line(image, start_point, end_point, cv::Scalar(0, 255, 0), 1);
    // }

    // // Display the image
    // cv::imshow("Dubins Path", image);
    // cv::waitKey(0);
}

int clamp(int value, int minVal, int maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

class Bezier {
public:
    Bezier() {}
    Bezier(const Eigen::Vector2f& p0_, const Eigen::Vector2f& t0_, const Eigen::Vector2f& p1_, const Eigen::Vector2f& t1_, float sharpness0_, float sharpness1_)
    :p0(p0_),t0(t0_),p1(p1_),t1(t1_),rotation0(0.0),rotation1(0.0),sharpness0(sharpness0_),sharpness1(sharpness1_) {}
    // Define a function to create a Bezier and calculate tangents
    void generateBezier(std::vector<Eigen::Vector2f>& points,
                        std::vector<Eigen::Vector2f>& tangents,
                        int numPoints, float no_norm = false) const {
        float scale = (p1 - p0).norm() / 3.0;
        Eigen::Vector2f r0(t0(0) * std::cos(rotation0) - t0(1) * std::sin(rotation0), t0(0) * std::sin(rotation0) + t0(1) * std::cos(rotation0));
        Eigen::Vector2f r1(t1(0) * std::cos(rotation1) - t1(1) * std::sin(rotation1), t1(0) * std::sin(rotation1) + t1(1) * std::cos(rotation1));
        Eigen::Vector2f c0 = p0;
        Eigen::Vector2f c1 = p0 + r0 / r0.norm() * scale * sharpness0;
        Eigen::Vector2f c2 = p1 - r1 / r1.norm() * scale * sharpness1;
        Eigen::Vector2f c3 = p1;

        for (int i = 0; i <= numPoints - 1; ++i) {
            float t = static_cast<float>(i) / (numPoints - 1);
            float u = 1.0f - t;

            Eigen::Vector2f point = 
                u * u * u * c0 +
                3 * u * u * t * c1 +
                3 * u * t * t * c2 +
                t * t * t * c3;

            Eigen::Vector2f tangent = 
                (-3 * u * u * c0 + 
                (3 * u * u - 6 * u * t) * c1 +
                (6 * u * t - 3 * t * t) * c2 +
                3 * t * t * c3);
            if (!no_norm) {
                tangent = tangent.normalized();
            }

            points.push_back(point);
            tangents.push_back(tangent);
        }
    }

    // Function to compute the loss and Jacobian
    void computeJacobianAndResidual(const std::vector<Eigen::Vector2f>& points,
                                const std::vector<Eigen::Vector2f>& tangents,
                                const std::vector<Eigen::Vector2f>& gtPoints,
                                const std::vector<Eigen::Vector2f>& gtTangents,
                                Eigen::MatrixXf& jacobian, Eigen::VectorXf& residual) const {
        int numPoints = points.size();
        jacobian = Eigen::MatrixXf::Zero(numPoints * 2, 5);
        residual = Eigen::VectorXf(numPoints * 2);

        float scale = (p1 - p0).norm() / 3.0;
        Eigen::Vector2f r0(t0(0) * std::cos(rotation0) - t0(1) * std::sin(rotation0), t0(0) * std::sin(rotation0) + t0(1) * std::cos(rotation0));
        Eigen::Vector2f r1(t1(0) * std::cos(rotation1) - t1(1) * std::sin(rotation1), t1(0) * std::sin(rotation1) + t1(1) * std::cos(rotation1));

        // Derivative with respect to p1
        Eigen::Matrix2f dc2_dp1 = Eigen::Matrix2f::Identity(); // Identity effect on c3 and c2
        Eigen::Matrix2f dc3_dp1 = Eigen::Matrix2f::Identity(); // Identity effect on c3 and c2
        // Derivative with respect to r1
        Eigen::Matrix2f dc2_dr1 = -scale * sharpness1 / r1.norm() * 
                                (Eigen::Matrix2f::Identity() - r1 * r1.transpose() / r1.squaredNorm());
        Eigen::Vector2f dr1_drotation1(- t1(0) * std::sin(rotation1) - t1(1) * std::cos(rotation1), t1(0) * std::cos(rotation1) - t1(1) * std::sin(rotation1));
        // Derivative with respect to sharpness0
        Eigen::Vector2f dc1_dsharpness0 = (r0 / r0.norm()) * scale;
        // Derivative with respect to sharpness1
        Eigen::Vector2f dc2_dsharpness1 = -(r1 / r1.norm()) * scale;

        for (int i = 0; i <= numPoints - 1; ++i) {
            Eigen::Vector2f resPoint = points[i] - gtPoints[i];
            Eigen::Vector2f resTangent = tangents[i] - gtTangents[i];
            float dRes_dPoint = 1;
            float dRes_dTangent = 1;

            float t = static_cast<float>(i) / (numPoints - 1);
            float u = 1.0f - t;

            float dPoint_dc1 = 3 * u * u * t;
            float dPoint_dc2 = 3 * u * t * t;
            float dPoint_dc3 = t * t * t;

            float dRawTangent_dc1 = 3 * u * u - 6 * u * t;
            float dRawTangent_dc2 = 6 * u * t - 3 * t * t;
            float dRawTangent_dc3 = 3 * t * t;

            Eigen::Vector2f rawTangent = tangents[i];
            Eigen::Matrix2f dTangent_dRawTangent = 1 / rawTangent.norm() * 
                                (Eigen::Matrix2f::Identity() - rawTangent * rawTangent.transpose() / rawTangent.squaredNorm());
            // Derivatives considering normalization
            Eigen::Matrix2f dTangent_dc1 = dTangent_dRawTangent * dRawTangent_dc1;
            Eigen::Matrix2f dTangent_dc2 = dTangent_dRawTangent * dRawTangent_dc2;
            Eigen::Matrix2f dTangent_dc3 = dTangent_dRawTangent * dRawTangent_dc3;

            // Update Jacobian
            // jacobian.block<2, 2>(i * 4, 0) = dRes_dPoint * dPoint_dc2 * dc2_dp1 + dRes_dPoint * dPoint_dc3 * dc3_dp1;
            // jacobian.block<2, 1>(i * 4, 2) = dRes_dPoint * dPoint_dc2 * dc2_dr1 * dr1_drotation1;
            // jacobian.block<2, 1>(i * 4, 4) = dRes_dPoint * dPoint_dc1 * dc1_dsharpness0;
            // jacobian.block<2, 1>(i * 4, 5) = dRes_dPoint * dPoint_dc2 * dc2_dsharpness1;
            jacobian.block<2, 2>(i * 2, 0) = dRes_dTangent * dTangent_dc2 * dc2_dp1 + dRes_dTangent * dTangent_dc3 * dc3_dp1;
            jacobian.block<2, 1>(i * 2, 2) = dRes_dTangent * dTangent_dc2 * dc2_dr1 * dr1_drotation1;
            jacobian.block<2, 1>(i * 2, 3) = dRes_dTangent * dTangent_dc1 * dc1_dsharpness0;
            jacobian.block<2, 1>(i * 2, 4) = dRes_dTangent * dTangent_dc2 * dc2_dsharpness1;

            // Update residual
            // residual.segment<2>(i * 2) = resPoint;
            residual.segment<2>(i * 2) = resTangent;
        }
    }

    Eigen::VectorXf computeDelta(const Eigen::MatrixXf& jacobian, const Eigen::VectorXf& residual) const {
        if (jacobian.hasNaN()) {
            std::cout << "jacobian contains NaN values." << std::endl;
        } else {
            std::cout << "jacobian does not contain NaN values." << std::endl;
        }
        if (residual.hasNaN()) {
            std::cout << "residual contains NaN values." << std::endl;
        } else {
            std::cout << "residual does not contain NaN values." << std::endl;
        }
        Eigen::MatrixXf JtJ = jacobian.transpose() * jacobian;
        Eigen::VectorXf Jtr = jacobian.transpose() * residual;

        // Eigen::VectorXf delta = JtJ.ldlt().solve(-Jtr);

        Eigen::VectorXf delta(Jtr.rows());
        float learningRate = 0.1;
        delta.segment<2>(0) = -learningRate * Jtr.segment<2>(0);
        delta(2) = -learningRate * Jtr(2);
        delta(3) = -learningRate * Jtr(3);
        delta(4) = -learningRate * Jtr(4);

        if (delta.hasNaN()) {
            std::cout << "delta contains NaN values." << std::endl;
        } else {
            std::cout << "delta does not contain NaN values." << std::endl;
        }
        return delta;
    }

    // Optimization function using Gaussian-Newton
    void optimizeBezierParameters(const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, float rotation_factor,
                                    int numPoints, int iterations) {
        std::cout << "Iteration " << "-" << ": p1=" << p1.transpose()
                << ", rotation1=" << rotation1
                << ", sharpness0=" << sharpness0
                << ", sharpness1=" << sharpness1 << std::endl;
                
        for (int iter = 0; iter < iterations; ++iter) {
            // Generate Bezier curve
            std::vector<Eigen::Vector2f> points, tangents;
            generateBezier(points, tangents, numPoints, true);

        int img_h = tangent_x.rows();
        int img_w = tangent_x.cols();
        std::vector<Grid> lineGrids;
        std::vector<Tangent> lineTangents;
        for (size_t i = 0; i < points.size(); ++i) {
            Grid g({static_cast<int>(points[i](0)), static_cast<int>(points[i](1))});
            Tangent t({tangents[i](0), tangents[i](1)});
            if (g.x >= 0 && g.x < img_h && g.y >= 0 && g.y < img_w) {
            } else {
                g.x = clamp(g.x, 0, img_h - 1);
                g.y = clamp(g.y, 0, img_w - 1);
            }
            lineGrids.push_back(g);
            lineTangents.push_back(t);
        }
        std::vector<Tangent> map_tangents;
        getMapTengent(lineGrids, tangent_x, tangent_y, rotation_factor, map_tangents);
        std::vector<Eigen::Vector2f> gtPoints;
        std::vector<Eigen::Vector2f> gtTangents;
        for (size_t i = 0; i < map_tangents.size(); ++i) {
            gtPoints.push_back({0., 0.});
            gtTangents.push_back({static_cast<float>(map_tangents[i].x), static_cast<float>(map_tangents[i].y)});
        }

            // Compute the loss and Jacobian here
            // Loss and Jacobian computation based on your specific loss function
            Eigen::MatrixXf jacobian;
            Eigen::VectorXf residual;
            computeJacobianAndResidual(points, tangents, gtPoints, gtTangents, jacobian, residual);
            // std::cout << "Jacobian: " << jacobian.transpose() << std::endl;
            // std::cout << "residual: " << residual.transpose() << std::endl;

            // Example: Placeholder for updating parameters
            Eigen::VectorXf delta = computeDelta(jacobian, residual);
            // std::cout << "Delta: " << delta.transpose() << std::endl;

            // Update parameters using delta
            // p1 += delta.segment<2>(0);
            // rotation1 += delta(2);
            // sharpness0 += delta(3);
            sharpness1 += delta(4);

            // Print/update for debugging
            float loss = residual.squaredNorm() / 2 / points.size();  // (x**2 + y**2) / 2
            std::cout << "Iteration " << iter << ": p1=" << p1.transpose()
                    << ", rotation1=" << rotation1
                    << ", sharpness0=" << sharpness0
                    << ", sharpness1=" << sharpness1 << ". Loss: " << loss << std::endl;
        }
    }
private:
    Eigen::Vector2f p0;
    Eigen::Vector2f t0;
    Eigen::Vector2f p1;
    Eigen::Vector2f t1;
    float rotation0;
    float rotation1;
    float sharpness0;
    float sharpness1;
};

void drawStraightLine(const Grid& A, const Grid& B, std::vector<Grid>& lineGrids, std::vector<Tangent>& lineTangents) {
    for (const auto& g : drawLineGrids(A, B)) {
        lineGrids.push_back(g);
    }
    const Tangent& lineTangent = drawLineTengent(A, B);
    for (size_t i = 0; i < lineGrids.size(); ++i) {
        lineTangents.push_back(lineTangent);
    }
}

void drawDubinsCurve(const Grid& A, const Grid& B, const Tangent& a, const Tangent& b, std::vector<Grid>& lineGrids, std::vector<Tangent>& lineTangents) {
    // dubins curve
    if (a.len() < 1e-6 || b.len() < 1e-6) {
        for (const auto& g : drawLineGrids(A, B)) {
            lineGrids.push_back(g);
        }
        const Tangent& lineTangent = drawLineTengent(A, B);
        for (size_t i = 0; i < lineGrids.size(); ++i) {
            lineTangents.push_back(lineTangent);
        }
    } else {
        double theta_a = std::atan2(a.y, a.x);
        double theta_b = std::atan2(b.y, b.x);
        // std::cout << "Grid: " << A << "; " << B << std::endl;
        // std::cout << "Tangent: " << a << "; " << b << std::endl;
        // std::cout << "theta: " << theta_a << "; " << theta_b << std::endl;
        double start[3] = {static_cast<double>(A.x), static_cast<double>(A.y), theta_a};
        double end[3] = {static_cast<double>(B.x), static_cast<double>(B.y), theta_b};
        double turning_radius = 5;
        std::vector<Eigen::Vector2f> points;
        std::vector<Eigen::Vector2f> tangents;
        drawDubinsPath(start, end, turning_radius, points, tangents);
        points.push_back({B.x, B.y});
        tangents.push_back({b.x, b.y});
        for (size_t i = 0; i < points.size(); ++i) {
            Grid g({static_cast<int>(points[i](0)), static_cast<int>(points[i](1))});
            Tangent t({tangents[i](0), tangents[i](1)});
            if (lineGrids.size() > 0 && lineGrids.back() == g){
                continue;
            }
            lineGrids.push_back(g);
            lineTangents.push_back(t);
        }
    }   
}

void drawBezierCurve(const Bezier& bezier, int n, std::vector<Grid>& lineGrids, std::vector<Tangent>& lineTangents) {
    std::vector<Eigen::Vector2f> points;
    std::vector<Eigen::Vector2f> tangents;
    bezier.generateBezier(points, tangents, n);
    for (size_t i = 0; i < points.size(); ++i) {
        Grid g({static_cast<int>(points[i](0)), static_cast<int>(points[i](1))});
        Tangent t({tangents[i](0), tangents[i](1)});
        if (lineGrids.size() > 0 && lineGrids.back() == g){
            continue;
        }
        lineGrids.push_back(g);
        lineTangents.push_back(t);
    }
}

void drawBezierCurve(const Bezier& bezier, const Grid& A, const Grid& B, const Tangent& a, const Tangent& b, std::vector<Grid>& lineGrids, std::vector<Tangent>& lineTangents) {
    if (a.len() < 1e-6 || b.len() < 1e-6) {
        for (const auto& g : drawLineGrids(A, B)) {
            lineGrids.push_back(g);
        }
        const Tangent& lineTangent = drawLineTengent(A, B);
        for (size_t i = 0; i < lineGrids.size(); ++i) {
            lineTangents.push_back(lineTangent);
        }
    } else {
        int n = 2*int(A.distance(B));
        drawBezierCurve(bezier, n, lineGrids, lineTangents);
    }
}

class Bridge {
private:
    std::vector<Grid> lineGrids;
    std::vector<Tangent> lineTangents;
    Bezier bezier;

public:
    Bridge() {}
    Bridge(const std::vector<Grid>& lineGrids_, const std::vector<Tangent>& lineTangents_): lineGrids(lineGrids_), lineTangents(lineTangents_) {}
    Bridge(const Grid& A, const Grid& B, const Tangent& a, const Tangent& b, int mode=0) {
        if (mode == 0) {
            drawStraightLine(A, B, lineGrids, lineTangents);
        } else if (mode == 1) {
            drawDubinsCurve(A, B, a, b, lineGrids, lineTangents);
        } else if (mode == 2) {
            bezier = Bezier(
                {static_cast<float>(A.x), static_cast<float>(A.y)}, 
                {static_cast<float>(a.x), static_cast<float>(a.y)}, 
                {static_cast<float>(B.x), static_cast<float>(B.y)}, 
                {static_cast<float>(b.x), static_cast<float>(b.y)}, 
                1, 
                1
            );
            drawBezierCurve(bezier, A, B, a, b, lineGrids, lineTangents);
        }
    }
    Bridge(const Grid& A, const Grid& B, const Tangent& a, const Tangent& b, float s0, float s1, int mode=2) {
        if (mode == 0) {
            // drawStraightLine(A, B, lineGrids, lineTangents, s0, s1);
        } else if (mode == 1) {
            // drawDubinsCurve(A, B, a, b, lineGrids, lineTangents, s0, s1);
        } else if (mode == 2) {
            bezier = Bezier(
                {static_cast<float>(A.x), static_cast<float>(A.y)}, 
                {static_cast<float>(a.x), static_cast<float>(a.y)}, 
                {static_cast<float>(B.x), static_cast<float>(B.y)}, 
                {static_cast<float>(b.x), static_cast<float>(b.y)}, 
                s0, 
                s1
            );
            drawBezierCurve(bezier, A, B, a, b, lineGrids, lineTangents);
        }
    }
    const std::vector<Grid>& getLineGrids() const {
        return lineGrids;
    }
    const std::vector<Tangent>& getLineTangents() const {
        return lineTangents;
    }
    void setLineGrids(const std::vector<Grid>& lineGrids_) {
        lineGrids = lineGrids_;
    }
    void setLineTangents(std::vector<Tangent>& lineTangents_) {
        lineTangents = lineTangents_;
    }
    const Bezier& getBezier() const {
        return bezier;
    }
    Bezier& getMutableBezier() {
        return bezier;
    }
    bool isCollisionFree(const Eigen::MatrixXi& traversability) const {
        if (lineGrids.size() <= 1 || lineGrids.front() == lineGrids.back()) { return false; }
        for (size_t i = 0; i < lineGrids.size(); i++) {
            if (traversability(lineGrids[i].x, lineGrids[i].y) == 0) {
                return false;
            }
        }
        return true;
    }
};

float calculateMean(const std::vector<float>& data) {
    float sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

float calculateCovDev(const std::vector<float>& data) {
    float mean = calculateMean(data);
    float sum = 0.0;
    for (const auto& value : data) {
        sum += (value - mean) * (value - mean);
    }
    return sum / data.size();
}

float calculateStdDev(const std::vector<float>& data) {
    return std::sqrt(calculateCovDev(data));
}

const float PI = 3.1415926;
float tangentDissimilarity(const std::vector<Tangent>& tangent_a, const std::vector<float>& conf_b, const std::vector<Tangent>& tangent_b, const std::vector<float>& discorage_b) {
    std::vector<float> xs;
    for (size_t i = 0; i < tangent_b.size(); i++) {
        float similarity = 0;
        float d = tangent_a[i].len()*tangent_b[i].len();
        float discrg = discorage_b[i];
        if (d > 0) {
            float cos = tangent_a[i].dot(tangent_b[i]) / d;
            float radian = acos(cos) / PI;
            // float radian = ((-0.698 * cos * cos - 0.872) * cos + 1.570) / PI;

            // float discrepency = 1 - (1 - radian) * d;
            // float discrepency = 1 - (1 - discrg);
            float discrepency = 1 - (1 - radian) * d * (1 - discrg);

            similarity = (1 - discrepency);
        }
        float conf = conf_b[i];
        float x = std::max((1 - similarity) * conf, (float)1e-6);
        xs.push_back(x);
    }

    return calculateMean(xs);
    // return calculateStdDev(xs);
}

double getIncCost(const Bridge& bridge) {
    // return parentPoint.distance(childPoint);
    return bridge.getLineGrids().size();
}

double getIncEnergy(const Bridge& bridge, const Eigen::MatrixXf& conf, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& discorage, float rotation_factor) {
    std::vector<float> map_confs;
    getMapConf(bridge.getLineGrids(), conf, map_confs);
    std::vector<Tangent> map_tangents;
    getMapTengent(bridge.getLineGrids(), tangent_x, tangent_y, rotation_factor, map_tangents);
    std::vector<float> map_discorages;
    getMapDiscorage(bridge.getLineGrids(), discorage, map_discorages);

    double consume = tangentDissimilarity(bridge.getLineTangents(), map_confs, map_tangents, map_discorages);
    return (0.5 - consume);
}

void optimizeBezier(Bridge& bridge, const Eigen::MatrixXf& conf, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& discorage, float rotation_factor) {
    bridge.getMutableBezier().optimizeBezierParameters(tangent_x, tangent_y, rotation_factor, 100, 100);
}

class NodeBridge {
public:
    Point parentPoint;
    Grid parentGrid;
    Point childPoint;
    Grid childGrid;
    Bridge bridge;
    float rotation_factor = 0.0;

    NodeBridge(const Point& childPoint_, const Grid& childGrid_)
     : parentPoint(Point({0, 0})), parentGrid(Grid({0, 0})), childPoint(childPoint_), childGrid(childGrid_) {}
    NodeBridge(const Point& parentPoint_, const Grid& parentGrid_, const Point& childPoint_, const Grid& childGrid_, const Tangent& parentTangent_, const Tangent& childTangent_)
     : parentPoint(parentPoint_), parentGrid(parentGrid_), childPoint(childPoint_), childGrid(childGrid_), bridge(parentGrid_, childGrid_, parentTangent_, childTangent_) {}
    NodeBridge(const Point& parentPoint_, const Grid& parentGrid_, const Point& childPoint_, const Grid& childGrid_, const Tangent& parentTangent_, const Tangent& childTangent_, float parentRatio_, float childRatio_)
     : parentPoint(parentPoint_), parentGrid(parentGrid_), childPoint(childPoint_), childGrid(childGrid_), bridge(parentGrid_, childGrid_, parentTangent_, childTangent_, parentRatio_, childRatio_) {}
    NodeBridge(const Point& parentPoint_, const Grid& parentGrid_, const Point& childPoint_, const Grid& childGrid_, const Bridge& bridge_)
     : parentPoint(parentPoint_), parentGrid(parentGrid_), childPoint(childPoint_), childGrid(childGrid_), bridge(bridge_) {}
    NodeBridge inv() const {
        std::vector<Grid> reversed_lineGrids(bridge.getLineGrids());
        std::reverse(reversed_lineGrids.begin(), reversed_lineGrids.end());
        std::vector<Tangent> reversed_tangents(bridge.getLineTangents());
        std::reverse(reversed_tangents.begin(), reversed_tangents.end());
        for (size_t i = 0; i < reversed_tangents.size(); ++i) {
            reversed_tangents[i].x = -reversed_tangents[i].x;
            reversed_tangents[i].y = -reversed_tangents[i].y;
        }
        return NodeBridge(childPoint, childGrid, parentPoint, parentGrid, Bridge(reversed_lineGrids, reversed_tangents));
    }
    DataItem getInc(const Eigen::MatrixXf& conf, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& discorage) const {
        double inc_cost = getIncCost(bridge);
        double inc_energy = getIncEnergy(bridge, conf, tangent_x, tangent_y, discorage, rotation_factor) * inc_cost;
        int inc_step = 1;
        return {inc_cost, inc_energy, inc_step};
    }
    bool isCollisionFree(const Eigen::MatrixXi& traversability) const {
        return bridge.isCollisionFree(traversability);
    }
    void optimizeBridge(const Eigen::MatrixXf& conf, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& discorage) {
        optimizeBezier(bridge, conf, tangent_x, tangent_y, discorage, rotation_factor);
    }
};

class NodeBridges {
private:
    std::vector<NodeBridge> nodeBridges;
public:
    NodeBridges(const Point& parentPoint_, const Grid& parentGrid_, const Point& childPoint_, const Grid& childGrid_, const Tangent& parentTangent_, const Tangent& childTangent_, int sampleCount) {
        for (int i = 0; i < sampleCount; ++i) {
            for (int j = 0; j < sampleCount; ++j) {
                for (int k = -sampleCount+1; k <= sampleCount-1; ++k) {
                    const Tangent& t0 = Tangent({-1, 0});
                    const Tangent& t1 = childTangent_.rotate(float(k) / sampleCount * (PI / 4));
                    auto nodeBridge = NodeBridge(parentPoint_, parentGrid_, childPoint_, childGrid_, t0, t1, 1 + float(i)/sampleCount, 1 + float(j)/sampleCount);
                    // nodeBridge.rotation_factor = float(k) / sampleCount * (PI / 4);
                    nodeBridges.emplace_back(nodeBridge);
                }
            }
        }
     }
    std::vector<NodeBridge> getNodeBridges() const {
        return nodeBridges;
    }
};

class Node {
public:
    Point point;
    Grid grid;
    int parent;
    std::vector<int> childrens;
    bool not_leaf;
    NodeBridge nb;
    DataItem inc;
    DataItem acc;
    Node(Point point_, Grid grid_, int parent_, std::vector<int> childrens_, bool not_leaf_, NodeBridge nb_, DataItem inc_, DataItem acc_) 
    : point(point_), grid(grid_), parent(parent_), childrens(childrens_), not_leaf(not_leaf_), nb(nb_), inc(inc_), acc(acc_)
    {}
    friend std::ostream& operator<<(std::ostream& os, const Node& dt);
};

std::ostream& operator<<(std::ostream& os, const Node& dt)
{
    os << "point x: " << dt.point.x << ", point y: " << dt.point.y << ", grid y: " << dt.grid.x << ", grid y: " << dt.grid.y << ", parent: " << dt.parent;
    for (size_t i = 0; i < dt.childrens.size(); ++i) {
        os << ", childrens[" << i << "]: " << dt.childrens[i];
    }
    os << ", not_leaf: " << dt.not_leaf;
    // os << ", nb: " << dt.nb;
    os << ", inc: " << dt.inc << ", acc: " << dt.acc;
    return os;
}

template <typename T>
std::vector<T> sample_random_elements(const std::vector<T>& vec, size_t num_samples) {
    // Copy the original vector to keep it unchanged
    std::vector<T> result(vec);

    // Obtain a time-based seed:
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed = 0;
    std::shuffle(result.begin(), result.end(), std::default_random_engine(seed));

    // Check if num_samples is greater than the vector's size
    if (num_samples > vec.size()) {
        num_samples = vec.size();
    }

    // Resize the result to contain only the first num_samples elements
    result.resize(num_samples);
    return result;
}

class Graph {
public:
    int V; // Number of vertices
    std::vector<std::vector<int>> adjMatrix;

    Graph(int V) : V(V), adjMatrix(V, std::vector<int>(V, 0)) {}

    void addEdge(int v, int w) {
        adjMatrix[v][w] = 1; // Add a directed edge from v to w
    }

    bool isCyclicUtil(int v, std::vector<bool>& visited, std::vector<bool>& recStack) {
        if (!visited[v]) {
            // Mark the current node as visited and part of recursion stack
            visited[v] = true;
            recStack[v] = true;

            // Recur for all vertices adjacent to this vertex
            for (int i = 0; i < V; ++i) {
                if (adjMatrix[v][i]) { // if there's a directed edge from v to i
                    if (!visited[i] && isCyclicUtil(i, visited, recStack))
                        return true;
                    else if (recStack[i])
                        return true;
                }
            }
        }

        // remove the vertex from recursion stack
        recStack[v] = false;
        return false;
    }

    bool isCyclic() {
        std::vector<bool> visited(V, false);
        std::vector<bool> recStack(V, false);

        // Call the recursive helper function to detect cycle in different DFS trees
        for (int i = 0; i < V; i++)
            if (isCyclicUtil(i, visited, recStack))
                return true;

        return false;
    }
};

class RRT {
public:
    RRT(Point input_start, double input_sampleRadius, double input_stepSize, int input_maxIter, double input_neighborRadius, int input_neighborCount, int input_extendMode, const Eigen::MatrixXf& input_conf, const Eigen::MatrixXf& input_discorage) :
        start(input_start),
        sampleRadius(input_sampleRadius),
        stepSize(input_stepSize),
        maxIter(input_maxIter),
        neighborRadius(input_neighborRadius),
        neighborCount(input_neighborCount),
        extendMode(input_extendMode) {

        // Seed the random number generator
        //std::srand(std::time(NULL));
        std::srand(0);

        conf = input_conf;
        discorage = input_discorage;
    }

    void sample_point(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& tangent2_x, const Eigen::MatrixXf& tangent2_y, int nearestNodeID, const Point& newPoint, const Grid& newGrid) {
        // In a real application, you would check for collisions here
        Node& nearestNode = nodes[nearestNodeID];
        const NodeBridge& nearestNodeBridge = NodeBridge(nearestNode.point, nearestNode.grid, newPoint, newGrid, {tangent_x(nearestNode.grid.x, nearestNode.grid.y), tangent_y(nearestNode.grid.x, nearestNode.grid.y)}, {tangent_x(newGrid.x, newGrid.y), tangent_y(newGrid.x, newGrid.y)});
        if (nearestNodeBridge.isCollisionFree(traversability)) {
            // clock_t c_start = clock();
            // Evaluate each nearby node to find the best parent
            const DataItem& incFromNearestNode = nearestNodeBridge.getInc(conf, tangent_x, tangent_y, discorage);
            const DataItem& accFromNearestNode = nearestNode.acc + incFromNearestNode;
            double flag_cost = accFromNearestNode.cost;
            double flag_energy = accFromNearestNode.energy;
            const std::vector<int>& nearByNodeIDs = getNearByNodeIDs(newPoint);
            const std::vector<std::pair<NodeBridge, int>>& nearByNodeBridges = createCollisionFreeNearbys(traversability, tangent_x, tangent_y, newPoint, newGrid, nearByNodeIDs, neighborCount);
            // clock_t c_nearBy = clock();
            int bestNearByNodeBridgeID = choose_parent(traversability, tangent_x, tangent_y, nearByNodeBridges, newPoint, newGrid, flag_cost, flag_energy);
            // If a better parent was found, update the parent and cost of the new node
            if (bestNearByNodeBridgeID >= 0) {
                int parentNodeID = nearByNodeBridges[bestNearByNodeBridgeID].second;
                Node& parentNode = nodes[parentNodeID];
                const NodeBridge& parentNodeBridge = nearByNodeBridges[bestNearByNodeBridgeID].first.inv();
                const DataItem& incFromParentNode = parentNodeBridge.getInc(conf, tangent_x, tangent_y, discorage);
                const DataItem& accFromParentNode = parentNode.acc + incFromParentNode;
                Node newNode = Node(newPoint, newGrid, parentNodeID, std::vector<int>(), false, parentNodeBridge, incFromParentNode, accFromParentNode);
                nodes.push_back(newNode);
                int newNodeID = nodes.size() - 1;
                connect_nodes(parentNodeID, newNodeID);
                // std::cout << "newNodeID: " << newNodeID << ", parentNodeID: " << parentNodeID << std::endl;
            } else {
                int parentNodeID = nearestNodeID;
                Node& parentNode = nearestNode;
                const NodeBridge& parentNodeBridge = nearestNodeBridge;
                const DataItem& incFromParentNode = incFromNearestNode;
                const DataItem& accFromParentNode = accFromNearestNode;
                Node newNode = Node(newPoint, newGrid, parentNodeID, std::vector<int>(), false, parentNodeBridge, incFromParentNode, accFromParentNode);
                nodes.push_back(newNode);
                int newNodeID = nodes.size() - 1;
                connect_nodes(parentNodeID, newNodeID);
                // std::cout << "newNodeID: " << newNodeID << ", parentNodeID: " << parentNodeID << std::endl;
            }
            // clock_t c_parent = clock();
            int newNodeID = nodes.size() - 1;
            rewire(traversability, tangent_x, tangent_y, nearByNodeBridges, newNodeID);
            // clock_t c_rewire = clock();
            // if (verbose) {
            //     std::cout << "nearByNodeIDs: " << nearByNodeIDs.size() << " nearByNodeBridges: " << nearByNodeBridges.size()
            //               << " core time: " << std::fixed 
            //               << (float)(c_nearBy-c_start)*1000/CLOCKS_PER_SEC << " " 
            //               << (float)(c_parent-c_nearBy)*1000/CLOCKS_PER_SEC << " " 
            //               << (float)(c_rewire-c_parent)*1000/CLOCKS_PER_SEC << std::endl;
            // }
        }
    }

    int init(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& tangent2_x, const Eigen::MatrixXf& tangent2_y) {
        auto img_h = traversability.rows();
        auto img_w = traversability.cols();
        Grid startGrid = start.toGrid(img_h, img_w);
        Node startNode = Node(start, startGrid, -1, std::vector<int>(), false, NodeBridge(start, startGrid), {0, 0, 0}, {0, 0, 0});
        nodes.push_back(startNode);
        return 0;
    }

    int extend(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& tangent2_x, const Eigen::MatrixXf& tangent2_y) {
        auto img_h = traversability.rows();
        auto img_w = traversability.cols();
        for (int i = 0; i < maxIter; i++) {
            Point randomPoint;
            while (true) {
                randomPoint = getRandomPoint();
                if (randomPoint.distance(start) > sampleRadius) {
                    continue;
                }
                Point newPoint = steerTowards(nodes[getNearestNodeID(randomPoint, img_h, img_w)].point, randomPoint, tangent2_x, tangent2_y, img_h, img_w);
                Grid newGrid = newPoint.toGrid(img_h, img_w);
                int nearestNodeID = getNearestNodeID(newPoint, img_h, img_w);
                const Node& nearestNode = nodes[nearestNodeID];
                const NodeBridge& nearestNodeBridge = NodeBridge(nearestNode.point, nearestNode.grid, newPoint, newGrid, {tangent_x(nearestNode.grid.x, nearestNode.grid.y), tangent_y(nearestNode.grid.x, nearestNode.grid.y)}, {tangent_x(newGrid.x, newGrid.y), tangent_y(newGrid.x, newGrid.y)});
                if (!nearestNodeBridge.isCollisionFree(traversability)) {
                    break;
                }
                sample_point(traversability, tangent_x, tangent_y, tangent2_x, tangent2_y, nearestNodeID, newPoint, newGrid);
                break;
            }
        }
        return 0;
    }

    int append(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Eigen::MatrixXf& tangent2_x, const Eigen::MatrixXf& tangent2_y, const std::vector<Point>& newPoints) {
        auto img_h = traversability.rows();
        auto img_w = traversability.cols();

        for (size_t i = 0; i < newPoints.size(); i++) {
            Point newPoint = newPoints[i];
            Grid newGrid = newPoint.toGrid(img_h, img_w);
            int nearestNodeID = getNearestNodeID(newPoint, img_h, img_w);
            sample_point(traversability, tangent_x, tangent_y, tangent2_x, tangent2_y, nearestNodeID, newPoint, newGrid);
        }
        return 0;
    }

    int build(Eigen::MatrixXf& result_map, int img_h, int img_w) {
        // Graph g(nodes.size());
        // for (size_t i = 0; i < nodes.size(); i++) {
        //     const auto& node = nodes[i];
        //     // std::cout << "node " << i << ": " << node << std::endl;
        //     if (node.parent != -1) {
        //         g.addEdge(node.parent, i);
        //     }
        // }
        // if (g.isCyclic())
        //     std::cout << "Graph contains a cycle" << std::endl;
        // else
        //     std::cout << "Graph does not contain a cycle" << std::endl;

        for (size_t i = 0; i < nodes.size(); i++) {
            // std::cout << i << "/" << nodes.size() << std::endl;
            const auto& node = nodes[i];
            if (node.parent != -1) {
                const std::vector<Grid>& lineGrids = node.nb.bridge.getLineGrids();
                for (size_t j = 1; j < lineGrids.size()-1; j++) {
                    const auto& lineGird = lineGrids[j];
                    int lineGird_idx = lineGird.x * img_w + lineGird.y;
                    result_map(lineGird_idx, 0) = 3;
                    update_dataitem(result_map, lineGird_idx, node);
                }
            }
            // std::cout << " node.parent " << node.parent << std::endl;
        }
        for (size_t i = 0; i < nodes.size(); i++) {
            const auto& node = nodes[i];
            const auto& grid = node.grid;
            int idx = grid.x * img_w + grid.y;
            if (node.not_leaf) {
                result_map(idx, 0) = 1;
            } else {
                result_map(idx, 0) = 2;
            }
            if (node.parent != -1) {
                const auto& parentGrid = nodes[node.parent].grid;
                result_map(idx, 1) = parentGrid.x;
                result_map(idx, 2) = parentGrid.y;
            } else {
                result_map(idx, 1) = -1;
                result_map(idx, 2) = -1;
            }
            update_dataitem(result_map, idx, node);
        }
        return 0;
    }
    
private:
    Point start;
    double sampleRadius;
    double stepSize;
    int maxIter;
    double neighborRadius;
    int neighborCount;
    int extendMode;  // 0 unknown, 1 cost, 2 consume
    std::vector<Node> nodes;
    Eigen::MatrixXf conf;
    Eigen::MatrixXf discorage;

    Point getRandomPoint() {
        return { (double)rand() / RAND_MAX, (double)rand() / RAND_MAX };  // shoud I consider 1?
    }

    int getNearestNodeID(const Point& point, int img_h, int img_w) {
        int nearestNodeID = -1;
        double nearestDist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < nodes.size(); i++) {
            double dist = point.distance(nodes[i].point);
            if (point.toGrid(img_h, img_w) == nodes[i].point.toGrid(img_h, img_w)) {  // sepecial section for nodes in same grid
                dist -= 1;
            }
            if (dist < nearestDist) {
                nearestNodeID = i;
                nearestDist = dist;
            }
        }
        return nearestNodeID;
    }

    std::vector<int> getNearByNodeIDs(const Point& point) {
        std::vector<int> nearby;
        for (size_t i = 0; i < nodes.size(); i++) {
            double dist = point.distance(nodes[i].point);
            if (dist < neighborRadius) {
                nearby.push_back(i);
            }
        }
        return nearby;
    }

    std::vector<std::pair<NodeBridge, int>> createCollisionFreeNearbys(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const Point& point, const Grid& grid, const std::vector<int>& nodeIDs, int sample_count = -1) {
        std::vector<std::pair<NodeBridge, int>> nearby;
        for (auto nodeID : nodeIDs) {
            const Node& n = nodes[nodeID];
            const NodeBridge& nodeBridge = NodeBridge(point, grid, n.point, n.grid, {tangent_x(grid.x, grid.y), tangent_y(grid.x, grid.y)}, {tangent_x(n.grid.x, n.grid.y), tangent_y(n.grid.x, n.grid.y)});
            if (nodeBridge.isCollisionFree(traversability)) {
                nearby.push_back(std::pair<NodeBridge, int>(nodeBridge, nodeID));
            }
        }
        if (sample_count < 0) {
            return nearby;
        } else {
            std::vector<int> samples;
            for (size_t i = 0; i < nearby.size(); ++i) {
                samples.push_back(i);
            }
            const auto& sampled = sample_random_elements(samples, sample_count);
            std::vector<std::pair<NodeBridge, int>> nearby_sampled;
            for (size_t i = 0; i < sampled.size(); ++i) {
                nearby_sampled.push_back(nearby[sampled[i]]);
            }
            return nearby_sampled;
            // return sample_random_elements(nearby, sample_count);
        }
    }

    Point steerTowards(Point from, Point to, const Eigen::MatrixXf& tangent2_x, const Eigen::MatrixXf& tangent2_y, int img_h, int img_w) {
        double dist = from.distance(to);
        Point result;
        if (dist < stepSize) {
            result = to;
        } else {
            double theta = std::atan2(to.y - from.y, to.x - from.x);
            result = { from.x + stepSize * cos(theta), from.y + stepSize * sin(theta) };
        }
        const Tangent& t = Tangent({result.x - from.x, result.y - from.y});
        // return { from.x + t.x, from.y + t.y };
        Grid from_grid = from.toGrid(img_h, img_w);
        const Tangent& ta = t.norm();
        const Tangent& tb = Tangent({tangent2_x(from_grid.x, from_grid.y), tangent2_y(from_grid.x, from_grid.y)});
        Tangent tc;
        if (ta.len() + tb.len() == 0) {
            tc = {0, 0};
        } else {
            tc = (ta + tb) / (ta.len() + tb.len());
        }
        return { from.x + t.len() * tc.x, from.y + t.len() * tc.y };
    }

    void breakup_nodes(int parentID, int childID) {
        nodes[parentID].childrens.erase(std::remove(nodes[parentID].childrens.begin(), nodes[parentID].childrens.end(), childID), nodes[parentID].childrens.end());
        nodes[parentID].not_leaf = nodes[parentID].childrens.size() != 0;
        nodes[childID].parent = -1;
    }

    void connect_nodes(int parentID, int childID) {
        nodes[parentID].childrens.push_back(childID);
        nodes[parentID].not_leaf = true;                    
        nodes[childID].parent = parentID;
    }

    void update_dataitem(Eigen::MatrixXf& result_map, int idx, const Node& node) {
        result_map(idx, 3) = node.acc.cost;
        result_map(idx, 4) = node.acc.energy;
        result_map(idx, 5) = node.acc.step;
        result_map(idx, 6) = node.inc.cost;
        result_map(idx, 7) = node.inc.energy;
    }

    int choose_parent(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, 
        const std::vector<std::pair<NodeBridge, int>>& nearByNodeBridges, const Point& newPoint, const Grid& newGrid, double flag_cost, double flag_energy) {
        int bestNearByNodeBridgeID = -1;
        double flag_consume = 0.5 * flag_cost - flag_energy;
        for (size_t i = 0; i < nearByNodeBridges.size(); ++i) {
            const auto& nearByNodeBridge = nearByNodeBridges[i];
            const auto& nodeBridge = nearByNodeBridge.first;
            const auto& nearByNodeID = nearByNodeBridge.second;
            // already collision free
            const auto& n = nodes[nearByNodeID];
            DataItem inc = nodeBridge.inv().getInc(conf, tangent_x, tangent_y, discorage);
            DataItem acc = n.acc + inc;
            if (extendMode == 1) {
                if (acc.cost < flag_cost) {
                    bestNearByNodeBridgeID = i;
                    flag_cost = acc.cost;
                }
            } else if (extendMode == 2) {
                double acc_consume = 0.5 * acc.cost - acc.energy;
                if (acc_consume < flag_consume) {
                    bestNearByNodeBridgeID = i;
                    flag_consume = acc_consume;
                }
            } else {
                return -1;
            }
        }
        return bestNearByNodeBridgeID;
    }
    
    // Function to rewire the tree with the new node
    void rewire(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, 
        const std::vector<std::pair<NodeBridge, int>>& nearByNodeBridges, int newNodeID) {
    	const Node& new_node = nodes[newNodeID];
        for (size_t i = 0; i < nearByNodeBridges.size(); ++i) {
            const auto& nearByNodeBridge = nearByNodeBridges[i];
            const auto& nodeBridge = nearByNodeBridge.first;
            const auto& nearByNodeID = nearByNodeBridge.second;
            // already collision free
            const auto& n = nodes[nearByNodeID];
            DataItem inc = nodeBridge.getInc(conf, tangent_x, tangent_y, discorage);
            DataItem acc = new_node.acc + inc;
            bool need = false;
            if (extendMode == 1) {
                need = acc.cost < n.acc.cost;
            } else if (extendMode == 2) {
                need = 0.5 * acc.cost - acc.energy < 0.5 * n.acc.cost - n.acc.energy;
            } else {
                need = false;
            }
            if (need) {
                int oldParentID = nodes[nearByNodeID].parent;
                int newParentID = newNodeID;
                if (oldParentID != -1) {  // if the nearby node has parent, then rewrite it
                    // std::cout << "nearByNodeID: " << nearByNodeID << ", oldParentID: " << oldParentID << ", newParentID: " << newParentID << std::endl;
                    // std::cout << "nodes[nearByNodeID]: " << nodes[nearByNodeID] << std::endl;
                    // std::cout << "nodes[oldParentID]: " << nodes[oldParentID] << std::endl;
                    // std::cout << "nodes[newParentID]: " << nodes[newParentID] << std::endl;

                    breakup_nodes(oldParentID, nearByNodeID);
                    connect_nodes(newParentID, nearByNodeID);

                    nodes[nearByNodeID].nb = nodeBridge;
                    nodes[nearByNodeID].inc = inc;
                    nodes[nearByNodeID].acc = acc;

                    // std::cout << "updated nodes[nearByNodeID]: " << nodes[nearByNodeID] << std::endl;
                    updateDescendant(nearByNodeID, tangent_x, tangent_y);
                }
            }
        }
    }

    void updateDescendant(int nodeID, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y) {
        for (auto childID : nodes[nodeID].childrens) {
            // std::cout << "nodeID: " << nodeID << ", acc: " << nodes[nodeID].acc << ", inc: " << nodes[nodeID].inc << std::endl;
            // std::cout << "childID: " << childID << ", acc: " << nodes[childID].acc << ", inc: " << nodes[childID].inc << std::endl;
            nodes[childID].acc = nodes[nodeID].acc + nodes[childID].inc;
            updateDescendant(childID, tangent_x, tangent_y); // Recursive call to update the entire subtree
        }
    }
};

Eigen::MatrixXf rrt(
    const Eigen::MatrixXi& traversability,     /* HxW ndarray */
    const Eigen::MatrixXf& conf,     /* HxW ndarray */
    const Eigen::MatrixXf& tangent_x,     /* HxW ndarray */
    const Eigen::MatrixXf& tangent_y,     /* HxW ndarray */
    const Eigen::MatrixXf& tangent2_x,     /* HxW ndarray */
    const Eigen::MatrixXf& tangent2_y,     /* HxW ndarray */
    const Eigen::MatrixXf& discorage,     /* HxW ndarray */
    const Eigen::VectorXi& start, /* 2 ndarray */
    const Eigen::MatrixXi& pros, /* Lx2 ndarray */
    const Eigen::MatrixXi& keys, /* Lx2 ndarray */
    double sampleRadius,
    double stepSize,
    int maxIter,
    double neighborRadius,
    int neighborCount,
    int extendMode
    )
{
    auto img_h = traversability.rows();
    auto img_w = traversability.cols();

    std::vector<Point> proPoints;
    for (ssize_t n = 0; n < pros.rows(); n++) {
        proPoints.push_back(Point{pros(n, 0) / float(img_h), pros(n, 1) / float(img_w)});
    }
    std::vector<Point> keyPoints;
    for (ssize_t n = 0; n < keys.rows(); n++) {
        keyPoints.push_back(Point{keys(n, 0) / float(img_h), keys(n, 1) / float(img_w)});
    }

    int ret = 0;
    RRT rrt(Point{start(0) / float(img_h), start(1) / float(img_w)}, sampleRadius, stepSize, maxIter, neighborRadius, neighborCount, extendMode, conf, discorage);
    ret = rrt.init(traversability, tangent_x, tangent_y, tangent2_x, tangent2_y);
    // std::cout << "extend: " << std::endl;
    ret = rrt.extend(traversability, tangent_x, tangent_y, tangent2_x, tangent2_y);
    // std::cout << "append: proPoints" << std::endl;
    ret = rrt.append(traversability, tangent_x, tangent_y, tangent2_x, tangent2_y, proPoints);
    // std::cout << "append: keyPoints" << std::endl;
    ret = rrt.append(traversability, tangent_x, tangent_y, tangent2_x, tangent2_y, keyPoints);
    Eigen::MatrixXf result_map = Eigen::MatrixXf::Zero(img_h * img_w, 8);  // 0: none, 1: not leaf, 2: leaf, 3: edge; parent_x; parent_y; cost; energy; step; inc_cost; inc_energy
    ret = rrt.build(result_map, img_h, img_w);
    return result_map;
}

class RNT {
public:
    RNT(Point input_start, double input_sampleRadius, int input_maxIter, int input_sampleCount, const Eigen::MatrixXf& input_conf, const Eigen::MatrixXf& input_discorage) :
        start(input_start),
        sampleRadius(input_sampleRadius),
        maxIter(input_maxIter),
        sampleCount(input_sampleCount) {

        // Seed the random number generator
        //std::srand(std::time(NULL));
        std::srand(0);

        conf = input_conf;
        discorage = input_discorage;
    }

    void sample_point(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, int nearestNodeID, const Point& newPoint, const Grid& newGrid, bool layze) {
        // In a real application, you would check for collisions here
        Node& nearestNode = nodes[nearestNodeID];
        const std::vector<NodeBridge>& nearestNodeBridges = NodeBridges(nearestNode.point, nearestNode.grid, newPoint, newGrid, {tangent_x(nearestNode.grid.x, nearestNode.grid.y), tangent_y(nearestNode.grid.x, nearestNode.grid.y)}, {tangent_x(newGrid.x, newGrid.y), tangent_y(newGrid.x, newGrid.y)}, sampleCount).getNodeBridges();
        for (const auto& nearestNodeBridge : nearestNodeBridges) {
            if (nearestNodeBridge.isCollisionFree(traversability)) {
                // clock_t c_start = clock();
                // Evaluate each nearby node to find the best parent
                DataItem incFromNearestNode;
                if (!layze) {
                    incFromNearestNode = nearestNodeBridge.getInc(conf, tangent_x, tangent_y, discorage);
                }
                const DataItem& accFromNearestNode = nearestNode.acc + incFromNearestNode;
                int parentNodeID = nearestNodeID;
                Node& parentNode = nearestNode;
                const NodeBridge& parentNodeBridge = nearestNodeBridge;
                const DataItem& incFromParentNode = incFromNearestNode;
                const DataItem& accFromParentNode = accFromNearestNode;
                Node newNode = Node(newPoint, newGrid, parentNodeID, std::vector<int>(), false, parentNodeBridge, incFromParentNode, accFromParentNode);
                nodes.push_back(newNode);
            }
        }
    }

    int init(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y) {
        auto img_h = traversability.rows();
        auto img_w = traversability.cols();
        Grid startGrid = start.toGrid(img_h, img_w);
        Node startNode = Node(start, startGrid, -1, std::vector<int>(), false, NodeBridge(start, startGrid), {0, 0, 0}, {0, 0, 0});
        nodes.push_back(startNode);
        return 0;
    }

    int extend(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, bool layze) {
        auto img_h = traversability.rows();
        auto img_w = traversability.cols();
        for (int i = 0; i < maxIter; i++) {
            Point randomPoint;
            while (true) {
                randomPoint = getRandomPoint();
                if (randomPoint.distance(start) > sampleRadius) {
                    continue;
                }
                int rootNodeID = 0;
                const Node& rootNode = nodes[rootNodeID];
                Point newPoint = randomPoint;
                Grid newGrid = newPoint.toGrid(img_h, img_w);
                // const NodeBridge& rootNodeBridge = NodeBridge(rootNode.point, rootNode.grid, newPoint, newGrid, {tangent_x(rootNode.grid.x, rootNode.grid.y), tangent_y(rootNode.grid.x, rootNode.grid.y)}, {tangent_x(newGrid.x, newGrid.y), tangent_y(newGrid.x, newGrid.y)}, 1);
                // if (!rootNodeBridge.isCollisionFree(traversability)) {
                //     break;
                // }
                sample_point(traversability, tangent_x, tangent_y, rootNodeID, newPoint, newGrid, layze);
                break;
            }
        }
        return 0;
    }

    int append(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, const std::vector<Point>& newPoints, bool layze) {
        auto img_h = traversability.rows();
        auto img_w = traversability.cols();

        for (size_t i = 0; i < newPoints.size(); i++) {
            Point newPoint = newPoints[i];
            Grid newGrid = newPoint.toGrid(img_h, img_w);
            int rootNodeID = 0;
            sample_point(traversability, tangent_x, tangent_y, rootNodeID, newPoint, newGrid, layze);
        }
        return 0;
    }

    void optimizeNode(const Eigen::MatrixXi& traversability, const Eigen::MatrixXf& tangent_x, const Eigen::MatrixXf& tangent_y, int node_id, bool update_inc = false) {
        nodes[node_id].nb.optimizeBridge(conf, tangent_x, tangent_y, discorage);
        std::vector<Grid> lineGrids;
        std::vector<Tangent> lineTangents;
        drawBezierCurve(nodes[node_id].nb.bridge.getBezier(), 100, lineGrids, lineTangents);
        nodes[node_id].nb.bridge.setLineGrids(lineGrids);
        nodes[node_id].nb.bridge.setLineTangents(lineTangents);
        nodes[node_id].nb.childGrid = lineGrids.back();
        // nodes[node_id].nb.childPoint = linepoints.back();
        if (update_inc) {
            nodes[node_id].inc = nodes[node_id].nb.getInc(conf, tangent_x, tangent_y, discorage);
        }
    }

    int build(Eigen::MatrixXf& result_map, int img_h, int img_w) {
        // Graph g(nodes.size());
        // for (size_t i = 0; i < nodes.size(); i++) {
        //     const auto& node = nodes[i];
        //     // std::cout << "node " << i << ": " << node << std::endl;
        //     if (node.parent != -1) {
        //         g.addEdge(node.parent, i);
        //     }
        // }
        // if (g.isCyclic())
        //     std::cout << "Graph contains a cycle" << std::endl;
        // else
        //     std::cout << "Graph does not contain a cycle" << std::endl;

        for (size_t i = 0; i < nodes.size(); i++) {
            // std::cout << i << "/" << nodes.size() << std::endl;
            const auto& node = nodes[i];
            if (node.parent != -1) {
                const std::vector<Grid>& lineGrids = node.nb.bridge.getLineGrids();
                for (size_t j = 1; j < lineGrids.size()-1; j++) {
                    const auto& lineGird = lineGrids[j];
                    int lineGird_idx = lineGird.x * img_w + lineGird.y;
                    result_map(lineGird_idx, 0) = 3;
                    update_dataitem(result_map, lineGird_idx, node);
                }
            }
            // std::cout << " node.parent " << node.parent << std::endl;
        }
        for (size_t i = 0; i < nodes.size(); i++) {
            const auto& node = nodes[i];
            const auto& grid = node.grid;
            int idx = grid.x * img_w + grid.y;
            if (node.not_leaf) {
                result_map(idx, 0) = 1;
            } else {
                result_map(idx, 0) = 2;
            }
            if (node.parent != -1) {
                const auto& parentGrid = nodes[node.parent].grid;
                result_map(idx, 1) = parentGrid.x;
                result_map(idx, 2) = parentGrid.y;
            } else {
                result_map(idx, 1) = -1;
                result_map(idx, 2) = -1;
            }
            update_dataitem(result_map, idx, node);
        }
        return 0;
    }
    
    const Node& at(int i) const {
        return nodes[i];
    }

    size_t size() const {
        return nodes.size();
    }

private:
    Point start;
    double sampleRadius;
    int maxIter;
    int sampleCount;
    std::vector<Node> nodes;
    Eigen::MatrixXf conf;
    Eigen::MatrixXf discorage;

    Point getRandomPoint() {
        return { (double)rand() / RAND_MAX, (double)rand() / RAND_MAX };  // shoud I consider 1?
    }

    void update_dataitem(Eigen::MatrixXf& result_map, int idx, const Node& node) {
        result_map(idx, 3) = node.acc.cost;
        result_map(idx, 4) = node.acc.energy;
        result_map(idx, 5) = node.acc.step;
        result_map(idx, 6) = node.inc.cost;
        result_map(idx, 7) = node.inc.energy;
    }
};

class RNTWrapper {
public:
    RNTWrapper(
    const Eigen::MatrixXi& traversability,     /* HxW ndarray */
    const Eigen::MatrixXf& conf,     /* HxW ndarray */
    const Eigen::MatrixXf& tangent_x,     /* HxW ndarray */
    const Eigen::MatrixXf& tangent_y,     /* HxW ndarray */
    const Eigen::MatrixXf& discorage,     /* HxW ndarray */
    const Eigen::VectorXi& start, /* 2 ndarray */
    const Eigen::MatrixXi& pros, /* Lx2 ndarray */
    const Eigen::MatrixXi& keys, /* Lx2 ndarray */
    double sampleRadius,
    int maxIter,
    int sampleCount,
    bool layze): 
    rnt(getPoint(start, traversability.rows(), traversability.cols()), sampleRadius, maxIter, sampleCount, conf, discorage),
    traversability_calling(traversability), 
    tangent_x_calling(tangent_x), 
    tangent_y_calling(tangent_y), 
    proPoints_calling(getPoints(pros, traversability.rows(), traversability.cols())), 
    keyPoints_calling(getPoints(keys, traversability.rows(), traversability.cols())),
    layze_calling(layze) {}
    Eigen::MatrixXf call() {
        auto img_h = traversability_calling.rows();
        auto img_w = traversability_calling.cols();

        int ret = 0;
        ret = rnt.init(traversability_calling, tangent_x_calling, tangent_y_calling);
        // std::cout << "extend: " << std::endl;
        ret = rnt.extend(traversability_calling, tangent_x_calling, tangent_y_calling, layze_calling);
        // std::cout << "append: proPoints" << std::endl;
        ret = rnt.append(traversability_calling, tangent_x_calling, tangent_y_calling, proPoints_calling, layze_calling);
        // std::cout << "append: keyPoints" << std::endl;
        ret = rnt.append(traversability_calling, tangent_x_calling, tangent_y_calling, keyPoints_calling, layze_calling);
        Eigen::MatrixXf result_map = Eigen::MatrixXf::Zero(img_h * img_w, 8);  // 0: none, 1: not leaf, 2: leaf, 3: edge; parent_x; parent_y; cost; energy; step; inc_cost; inc_energy
        ret = rnt.build(result_map, img_h, img_w);
        return result_map;
    }
    std::tuple<Eigen::MatrixXi, double> trace_id(int node_id) {
        const auto& acc = rnt.at(node_id).acc;
        double acc_consume = 0.5 * acc.cost - acc.energy;
        std::vector<Grid> vec;
        vec.push_back(rnt.at(node_id).grid);
        for (int i = node_id; i != -1; ) {
            const auto& node = rnt.at(i);
            const std::vector<Grid>& lineGrids = node.nb.bridge.getLineGrids();
            for (int j = int(lineGrids.size()) - 2; j >= 0; --j) {
                vec.push_back(lineGrids[j]);
            }
            i = node.parent;
        }
        Eigen::MatrixXi result_vec(vec.size(), 2);
        for (size_t i = 0; i < vec.size(); ++i) {
            result_vec(i, 0) = vec[i].x;
            result_vec(i, 1) = vec[i].y;
        }
        return std::make_tuple(result_vec, acc_consume);
    }
    std::tuple<Eigen::MatrixXi, double> trace_id_optimize(int node_id) {
        rnt.optimizeNode(traversability_calling, tangent_x_calling, tangent_y_calling, node_id, true);
        return trace_id(node_id);
    }
    std::tuple<Eigen::MatrixXi, double> trace_back(const Eigen::VectorXi& end) {
        int node_id = -1;
        for (size_t i = 0; i < rnt.size(); ++i) {
            if (rnt.at(i).grid.x == end(0) && rnt.at(i).grid.y == end(1)) {
                node_id = i;
            }
        }
        return trace_id(node_id);
    }
    std::tuple<Eigen::MatrixXi, double> trace_best() {
        int node_id = -1;
        // double flag_cost = std::numeric_limits<double>::max();
        double flag_consume = std::numeric_limits<double>::max();
        for (size_t i = 1; i < rnt.size(); ++i) {
            const auto& acc = rnt.at(i).acc;
            // double acc_cost = acc.cost;
            double acc_consume = 0.5 * acc.cost - acc.energy;
            // if (acc_cost < flag_cost) {
            //     node_id = i;
            //     flag_cost = acc_cost;
            // }
            if (acc_consume < flag_consume) {
                node_id = i;
                flag_consume = acc_consume;
            }
        }
        return trace_id(node_id);
    }
    std::tuple<Eigen::MatrixXi, double> trace_best_optimize() {
        int node_id = -1;
        // double flag_cost = std::numeric_limits<double>::max();
        double flag_consume = std::numeric_limits<double>::max();
        for (size_t i = 1; i < rnt.size(); ++i) {
            const auto& acc = rnt.at(i).acc;
            // double acc_cost = acc.cost;
            double acc_consume = 0.5 * acc.cost - acc.energy;
            // if (acc_cost < flag_cost) {
            //     node_id = i;
            //     flag_cost = acc_cost;
            // }
            if (acc_consume < flag_consume) {
                node_id = i;
                flag_consume = acc_consume;
            }
        }
        rnt.optimizeNode(traversability_calling, tangent_x_calling, tangent_y_calling, node_id, true);
        return trace_id(node_id);
    }
    std::tuple<Eigen::MatrixXi, double> trace_oricle(const Eigen::MatrixXi& oricle) {
        int img_h = traversability_calling.rows();
        int img_w = traversability_calling.cols();
        std::vector<Eigen::MatrixXf> distances;
        for (int ii = 0; ii < oricle.rows(); ++ii) {
            distances.push_back(Eigen::MatrixXf(img_h, img_w));
            for (int i = 0; i < img_h; ++i) {
                for (int j = 0; j < img_w; ++j) {
                    distances.back()(i, j) = std::sqrt(std::pow(float(i) - oricle(ii, 0), 2) + std::pow(float(j) - oricle(ii, 1), 2));
                }
            }
        }

        int node_id = -1;
        double flag_distance = std::numeric_limits<double>::max();
        Eigen::MatrixXi flag_result;
        for (size_t i = 1; i < rnt.size(); ++i) {
            const auto& res_trace_id = trace_id(i);
            const auto& result_vec = std::get<0>(res_trace_id);
            double acc_dist = 0;
            for (size_t j = 0; j < distances.size(); ++j) {
                double min_dist = std::numeric_limits<double>::max();
                for (int k = 0; k < result_vec.rows(); ++k) {
                    double dist = distances[j](result_vec(k, 0), result_vec(k, 1));
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
                acc_dist += min_dist;
            }
            if (acc_dist < flag_distance) {
                node_id = i;
                flag_distance = acc_dist;
                flag_result = result_vec;
            }
        }
        return std::make_tuple(flag_result, flag_distance);
    }
    std::tuple<Eigen::MatrixXi, double> trace_oricle_optimize(const Eigen::MatrixXi& oricle) {
        int img_h = traversability_calling.rows();
        int img_w = traversability_calling.cols();
        std::vector<Eigen::MatrixXf> distances;
        for (int ii = 0; ii < oricle.rows(); ++ii) {
            distances.push_back(Eigen::MatrixXf(img_h, img_w));
            for (int i = 0; i < img_h; ++i) {
                for (int j = 0; j < img_w; ++j) {
                    distances.back()(i, j) = std::sqrt(std::pow(float(i) - oricle(ii, 0), 2) + std::pow(float(j) - oricle(ii, 1), 2));
                }
            }
        }

        int node_id = -1;
        double flag_distance = std::numeric_limits<double>::max();
        Eigen::MatrixXi flag_result;
        for (size_t i = 1; i < rnt.size(); ++i) {
            const auto& res_trace_id = trace_id(i);
            const auto& result_vec = std::get<0>(res_trace_id);
            double acc_dist = 0;
            for (size_t j = 0; j < distances.size(); ++j) {
                double min_dist = std::numeric_limits<double>::max();
                for (int k = 0; k < result_vec.rows(); ++k) {
                    double dist = distances[j](result_vec(k, 0), result_vec(k, 1));
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
                acc_dist += min_dist;
            }
            if (acc_dist < flag_distance) {
                node_id = i;
                flag_distance = acc_dist;
                flag_result = result_vec;
            }
        }
        rnt.optimizeNode(traversability_calling, tangent_x_calling, tangent_y_calling, node_id);
        const auto& res_trace_id = trace_id(node_id);
        const auto& result_vec = std::get<0>(res_trace_id);
        const auto& acc_consume = std::get<1>(res_trace_id);
        double acc_dist = 0;
        for (size_t j = 0; j < distances.size(); ++j) {
            double min_dist = std::numeric_limits<double>::max();
            for (int k = 0; k < result_vec.rows(); ++k) {
                double dist = distances[j](result_vec(k, 0), result_vec(k, 1));
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            acc_dist += min_dist;
        }
        return std::make_tuple(result_vec, acc_dist);
    }
private:
    RNT rnt;
    Eigen::MatrixXi traversability_calling;
    Eigen::MatrixXf tangent_x_calling;
    Eigen::MatrixXf tangent_y_calling;
    std::vector<Point> proPoints_calling;
    std::vector<Point> keyPoints_calling;
    bool layze_calling;

    Point getPoint(const Eigen::VectorXi& mat, int img_h, int img_w) {
        return {mat(0) / float(img_h), mat(1) / float(img_w)};
    }
    std::vector<Point> getPoints(const Eigen::MatrixXi& mat, int img_h, int img_w) {
        std::vector<Point> points;
        for (ssize_t n = 0; n < mat.rows(); n++) {
            points.push_back(getPoint(mat.row(n), img_h, img_w));
        }
        return points;
    }
};

PYBIND11_MODULE(planning, m)
{
    m.doc() = "Applying planning in 2D Matrix";

    m.def("dijkstra",
        &dijkstra,
        py::arg("traversability"),
        py::arg("starts"),
        py::arg("directions"),
        py::arg("costs"));
        
    m.def("randomwalk",
        &randomwalk,
        py::arg("traversability"),
        py::arg("starts"),
        py::arg("directions"),
        py::arg("costs"));
        
    m.def("rrt",
        &rrt,
        py::arg("traversability"),
        py::arg("conf"),
        py::arg("tangent_x"),
        py::arg("tangent_y"),
        py::arg("tangent2_x"),
        py::arg("tangent2_y"),
        py::arg("discorage"),
        py::arg("start"),
        py::arg("pros"),
        py::arg("keys"),
        py::arg("sampleRadius"),
        py::arg("stepSize"),
        py::arg("maxIter"),
        py::arg("neighborRadius"),
        py::arg("neighborCount"),
        py::arg("extendMode"));

    py::class_<RNTWrapper>(m, "RNTWrapper")
        .def(py::init<
            Eigen::MatrixXi,     /* HxW ndarray */
            Eigen::MatrixXf,     /* HxW ndarray */
            Eigen::MatrixXf,     /* HxW ndarray */
            Eigen::MatrixXf,     /* HxW ndarray */
            Eigen::MatrixXf,     /* HxW ndarray */
            Eigen::VectorXi, /* 2 ndarray */
            Eigen::MatrixXi, /* Lx2 ndarray */
            Eigen::MatrixXi, /* Lx2 ndarray */
            double,
            int,
            int,
            bool>()
            ,
            py::arg("traversability"),
            py::arg("conf"),
            py::arg("tangent_x"),
            py::arg("tangent_y"),
            py::arg("discorage"),
            py::arg("start"),
            py::arg("pros"),
            py::arg("keys"),
            py::arg("sampleRadius"),
            py::arg("maxIter"),
            py::arg("sampleCount"),
            py::arg("layze")
        )
        .def("call", &RNTWrapper::call)
        .def("trace_id", &RNTWrapper::trace_id, py::arg("node_id"))
        .def("trace_id_optimize", &RNTWrapper::trace_id_optimize, py::arg("node_id"))
        .def("trace_back", &RNTWrapper::trace_back, py::arg("end"))
        .def("trace_best", &RNTWrapper::trace_best)
        .def("trace_best_optimize", &RNTWrapper::trace_best_optimize)
        .def("trace_oricle", &RNTWrapper::trace_oricle, py::arg("oricle"))
        .def("trace_oricle_optimize", &RNTWrapper::trace_oricle_optimize, py::arg("oricle"));
}
