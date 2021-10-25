#ifndef LINEALREGRESION_H
#define LINEALREGRESION_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

class LinealRegresion
{
public:
    LinealRegresion()
    {}

    float FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones);
};

#endif // LINEALREGRESION_H
