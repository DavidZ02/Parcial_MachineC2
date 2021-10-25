#include "linealregresion.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* Se necesita entrenar el modelo, lo que implica minimizar alguna funcion de costo
 * y de esta forma se puede medir la precision de la funcion de hipotesis. La funcion
 * de costo es la forma de penalizar al modelos por coeter un error
*/

float LinealRegresion::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X * theta -y).array(),2);

    return (diferencia.sum()/(2*X.rows()));
}


/* Se implementa la funcion para dar al algoritmo los valores de theta inciales,
 * que cambiaran iterativamente hasta que converga al valor minimo de la función
 * de costo. Basicamente describira el gradiente descendiente: El cual es dado por
 * la derivada parcial de la funcion */

std::tuple<Eigen::VectorXd, std::vector<float>> LinealRegresion::gradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones){
    /* Almacenamiento temporal para los valores theta */
    Eigen::MatrixXd temporal = theta;

    /* Variable con la cantidad de parametros m (Features) */
    int parametros = theta.rows();

    /*Ubicar el costo inicial, que se actualizar iterativamnete con los pesos */
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X,y,theta));

    /* Por cada iteracion se calcula la funcion error */
    for(int i = 0; i < iteraciones; i++){
        Eigen::MatrixXd error = X*theta-y;
            for(int j = 0; j < parametros; ++j){
                Eigen::MatrixXd X_i = X.col(j);
                Eigen::MatrixXd termino = error.cwiseProduct(X_i);
                temporal(j,0) = theta(j,0) - (alpha/X.rows())*termino.sum();
            }
            theta = temporal;
            costo.push_back(FuncionCosto(X,y,theta));
    }

    return std::make_tuple(theta, costo);
}
