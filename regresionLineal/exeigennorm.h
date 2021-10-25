#ifndef EXEIGENNORM_H
#define EXEIGENNORM_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

class exEigenNorm
{
    /*Para el constructor se necesitaran tres argumentos de entrada (atributos). */
    /*1. Nombre del dataset */
    std::string setDatos;

    /*2. Separador de columnas */
    std::string delimitador;

    /*3. Tiene o no cabecera*/
    bool header;

public:
    exEigenNorm(std::string datos, std::string separador, bool head):
        setDatos(datos),
        delimitador(separador),
        header(head){}
    std::vector<std::vector<std::string>> LeerCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int col);
    auto Promedio(Eigen::MatrixXd datos)->decltype(datos.colwise().mean());
    auto Desviacion(Eigen::MatrixXd datos)->decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd datos, float size_t);
    void VectorToFile(std::vector<float> vector, std::string nombre);
    void EigenToFile(Eigen::MatrixXd datos, std::string nombre);
};

#endif // EXEIGENNORM_H
