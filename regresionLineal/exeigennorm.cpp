#include "exeigennorm.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

//Primera funcion: Lectura de ficheros csv
//Vector de vectores String
//La idea es leer linea por linea y almacenar en un vector de vectores tipo String

std::vector<std::vector<std::string>> exEigenNorm::LeerCSV(){

    //Se abre el archivo para lectura solamente
    std::ifstream Archivo (setDatos);

    //Vector de vectores de tipo string que tendrá los datos del dataset
    std::vector<std::vector<std::string>> datosString;

    //Se itera a traves de cada linea del dataSet y se divide el contenido
    //dado por el delimitador provisto por el constructor
    std::string linea="";
    while(getline(Archivo,linea)){
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    //Se cierra el fichero (Archivo)
    Archivo.close();
    /*
     * Se retorna el vector de vectores de tipo string
     */
    return datosString;
}

/*
 * Se crea la segunda funcion para guardar el vector de vectores del tipo string
 * a una matriz Eigen. Similar a Pandas(Python) para presentar un dataFrame
 */
Eigen::MatrixXd exEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int col){
    /*
     * Si tiene cabecera la removemos
     */
    if(header==true){
        filas -= 1;
    }
    /*
     * Se itera sobre filas y columnas para almacenar en la matriz vacia(Tamaño filas*columnas), que basicamente
     * almacenará string en un vector: Luego lo pasaremos a float para ser manipulados
     */
    Eigen::MatrixXd dfMatriz(col, filas);
    for(int i=0 ; i< filas ; i++){
        for(int j=0 ; j< col ; j++){
            dfMatriz(j,i) = atof(datosString[i][j].c_str());
        }
    }
    /*
     * Se transpone la matriz para tener filas por columnas
    */
    return dfMatriz.transpose();
}

/*A continuación se van a implementar las funciones para la normalización
 *en c++, la palabra clave auto especifica que el tipo de la variable
 * que se empieza a declarar se deducirá automáticamente de su inicializador
 * y, para las funciones si su tipo de retorno es auto se evaluará
 * mediante la expresión del tipo de retorno en tiempo de ejecución
 * (inicializar variables sin especificar el tipo)
 */

/*
auto ExEigenNorm::Promedio(Eigen::MatriXd datos){
    * Se ingresa como entrada la matriz de datos (Datos) y regresa el promedio

    *return datos.colwise().mean();
}*/

/* NO se sabe que retorna datos.colwise().mean()
 * en c++ la herencia del tipo de dato no es directa o no se sabe que tipo de dato
 * debe retornar, entonces para ello se declara el tipo en una expresion de 'declctype'
 * con el fin de tener seguridad de que tipo de dato retornará la función */


auto exEigenNorm::Promedio(Eigen::MatrixXd Datos)->decltype(Datos.colwise().mean()){
    /* Se ingresa como entrada la matriz de datos (Datos) y regresa el promedio*/

    return Datos.colwise().mean();
}

/* Para implementar la función de desviación estandar
 * datos=x_i -promedio(x) */

auto exEigenNorm::Desviacion(Eigen::MatrixXd Datos) -> decltype(((Datos.array().square().colwise().sum())/(Datos.rows()-1)).sqrt()){
    return ((Datos.array().square().colwise().sum())/(Datos.rows())).sqrt();
}

/*Normalización Z-Score es una estratedia de normalización de datos
 * Que evita el problema de los outliers*/

Eigen::MatrixXd exEigenNorm::Normalizador(Eigen::MatrixXd datos){
    //auto promedio = Promedio(datos);

    Eigen::MatrixXd diferenciaPromedio = datos.rowwise() - Promedio(datos);  //(x_i  - promedio)

    //auto desviacion = Desviacion(diferenciaPromedio);

    Eigen::MatrixXd matrizNormalizada = diferenciaPromedio.array().rowwise() / Desviacion(diferenciaPromedio);

    return matrizNormalizada;
}

/* A continuacion se hara una funcion para dividir los datos en conjunto de datos
 * de entrenamiento y conjunto de datos de prueba
 */

//columnas 11 caracteristicas variable independiente (X)

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> exEigenNorm::TrainTestSplit(Eigen::MatrixXd Datos,float sizeTrain){
   int filas=Datos.rows();
   int filasTrain=round(sizeTrain*filas);
   int filasTest=filas-filasTrain;

   //Con Eigen se puede especificar el bloque de una matriz, por ejemplo se
   //pueden seleccionar las filas superiores para el conjunto de entrenamiento
   //indicando cuantas filas se desean,se selecciona desde 0 (fila 0) hasta el num de filas indicado

   Eigen::MatrixXd entrenamiento=Datos.topRows(filasTrain);

   /*Seleccionadas las filas superiores para entrenamiento se seleccionan las 11
    * primeras columnas (columnas a la izquierda) que representan las variables
    * independientes FEATURES
    */

   Eigen::MatrixXd X_train= entrenamiento.leftCols(Datos.cols()-1);

   //Se selecciona la variable dependiente que corresponde a la ultima columna
   Eigen::MatrixXd y_train= entrenamiento.rightCols(1);

   //Se realiza lo mismo para el conjunto de pruebas

   Eigen::MatrixXd prueba=Datos.bottomRows(filasTest);
   Eigen::MatrixXd X_test= prueba.leftCols(Datos.cols()-1);
   Eigen::MatrixXd y_test= prueba.rightCols(1);

   //finalmente se retorna una tupla con los datos de prueba y entrenamiento

   return std::make_tuple(X_train, y_train, X_test, y_test);

}

void exEigenNorm::VectorToFile(std::vector<float> vector, std::string nombre){
    std::ofstream fichero(nombre);
    std::ostream_iterator<float> iterador(fichero, "\n");
    std::copy(vector.begin(),vector.end(),iterador);
}

void exEigenNorm::EigenToFile(Eigen::MatrixXd datos, std::string nombre){
    std::ofstream fichero(nombre);
    if(fichero.is_open()){
        fichero << datos << "\n";
    }
}
