#include "LagrangeInterp.h";
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include "LagrangeInterp.h";
#include "LinearRegression.h";


// Para conseguir usar caractéres Unicode nos prints
#ifdef _WIN32
#include <windows.h>
#endif

void runLagrange() {
    /*
    para printar caractéres unicode.
    */
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    std::vector<double> xVal = { 1.0, 2.0, 3.0 };
    std::vector<double> yVal = { 2.0, 1.0, 2.0 };
    double a = 1.5;


    double result = LagrangeInterp(xVal, yVal, a);
    LagrageInterpBari<double> interpol(xVal, yVal);
    double resultBari = interpol.evaluate(a);


    std::cout << "Resultados 1ª func:\n" << result << "\n\nResultados 2ª func:\n" << resultBari << "\n";
};

void runRegression() {

};

int main() {
    runLagrange();
	return 0;
};