#include "exeigennorm.h"
#include "linealregresion.h"

#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

/*
 *En primer lugar se creara una clase llamada exEigenNorm, la cual nos permitira
 *leer un dataset, extraer los datos, montar sobre la estructura Eigen para
 *normalizar los datos
 */

int main(int argc, char *argv[])
{
    /* Se crea un objeto del tipo exEigenNorm, se incluye los tres argumentos del constructor:
     * nombre del dataset, delimitador, flag(tiene o no tiene header) */
    exEigenNorm extraccion(argv[1],argv[2],argv[3]);
    LinealRegresion Lr;

    /* Se leen los datos del archivo, por la funcion leerCSV() */
    std::vector<std::vector<std::string>> dataFrame = extraccion.LeerCSV();

    /* Para probar la segunda funcion CSVtoEigen() se define la cantidad de filas y columnas
       basados en los datos de columna */
    int filas = dataFrame.size()+1;
    int columnas = dataFrame[0].size();
    Eigen::MatrixXd matrizDataF = extraccion.CSVtoEigen(dataFrame, filas, columnas);

    /* Para desarrollar el primer algoritmo de regresion lineal, en donde se probara con los datos de lo vinos(winedata.csv)
     * se presentara la regresion lineal para multiples variables. Entonces dada la naturaleza de la regresion lienal, si se
     * tiene variables con diferentes unidades, una variable podria beneficiar/estropear otra variable; se necesitara estandarizar
     * los datos, dejando a todas las variables del mismo orden de magnitud y centradas en cero. Para ello se contruira una funcion
     * de normalizacion basada en el set score normalizacion. se necesitan tres funciones: la funcion de normalizacion,
     * la del promedio y la desviacion estandar. */

    Eigen::MatrixXd matrizNormalizada=extraccion.Normalizador(matrizDataF);
    std::cout<<"La matriz normalizada es:\n\n "<<matrizNormalizada<<std::endl;

    Eigen::MatrixXd Promedio=extraccion.Promedio(matrizDataF);
    std::cout<<"\nEl promedio es: \n\n"<<Promedio<<std::endl;

    Eigen::MatrixXd diferenciaPromedio = matrizDataF.rowwise() - extraccion.Promedio(matrizDataF);

    Eigen::MatrixXd DesviacionE=extraccion.Desviacion(diferenciaPromedio);
    std::cout<<"\nLa desviación estandar es: \n\n"<<DesviacionE<<std::endl<<std::endl;

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> divDatos= extraccion.TrainTestSplit(matrizNormalizada, 0.8);
    Eigen::MatrixXd X_Train,y_Train, X_Test, y_Test;
    std::tie(X_Train,y_Train,X_Test,y_Test)=divDatos;

    //Inspeccion visual de la división de los datos de entrenamiento y prueba

        std::cout<< "Variable independiente (feature): "<<std::endl;
        std::cout<< "Tamaño original: "<<matrizNormalizada.rows()<<std::endl;
        std::cout<< "Tamaño entrenamiento(filas): "<<X_Train.rows()<<std::endl;
        std::cout<< "Tamaño entrenamiento (columnas): "<<X_Train.cols()<<std::endl;
        std::cout<< "Tamaño prueba (filas): "<<X_Test.rows()<<std::endl;
        std::cout<< "Tamaño prueba (columnas): "<<X_Test.cols()<<std::endl<<std::endl;

        std::cout<< "Variable dependiente (target): "<<std::endl;
        std::cout<< "Tamaño original: "<<matrizNormalizada.rows()<<std::endl;
        std::cout<< "Tamaño entrenamiento(filas): "<<y_Train.rows()<<std::endl;
        std::cout<< "Tamaño entrenamiento (columnas): "<<y_Train.cols()<<std::endl;
        std::cout<< "Tamaño prueba (filas): "<<y_Test.rows()<<std::endl;
        std::cout<< "Tamaño prueba (columnas): "<<y_Test.cols()<<std::endl<<std::endl;

    /* A continuacion se procede a probar la clase de regresion lineal*/

    Eigen::VectorXd vectorEntrenamiento = Eigen::VectorXd::Ones(X_Train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_Test.rows());
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_Train.rows());

    /* Redimension de las matrices para ubicacion en los valores de ONES (Similar a reshape con Numpy) */
    X_Train.conservativeResize(X_Train.rows(), X_Train.cols() + 1);
    X_Train.col(X_Train.cols() - 1) = vectorTrain;

    X_Test.conservativeResize(X_Test.rows(), X_Test.cols() + 1);
    X_Test.col(X_Test.cols() - 1) = vectorTest;


    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_Train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    Eigen::VectorXd thetaOut;
    std::vector<float> costo;

    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteD = Lr.gradienteDescendiente(X_Train, y_Train, theta, alpha, iteraciones);
    std::tie(thetaOut, costo) = gradienteD;

    std::cout<<"\nTheta: " << thetaOut << std::endl;
    std::cout<<"\nCosto: \n " << std::endl;

    for(auto valor:costo){
        std::cout<< valor << std::endl;
    }

    extraccion.VectorToFile(costo,"Costo.txt");
    extraccion.EigenToFile(thetaOut,"ThetaOut.txt");

    return EXIT_SUCCESS;
}
