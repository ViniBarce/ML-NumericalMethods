#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>

// 1. Para conseguir usar caractéres Unicode nos prints
#ifdef _WIN32
#include <windows.h>
#endif

/*
    Essa é a implementação básica de Lagrange Interp.
    porém a checagem do vetor x é de O(n^2), então não é a função ideal para plots.
*/
template <typename T>
double LagrangeInterp(const std::vector<T>& x, const std::vector<T>& y, const double& a) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x e y precisam ser do mesmo tamanho.");
    }
    if (x.empty()) {
        return 0.0;
    }

    size_t n = x.size();
    /* 
        checar se tem divisão por zero, checagem básica porém suficiente.
    */
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = i + 1; k < n; ++k) {
            const T& val1 = x[i];
            const T& val2 = x[k];

            if (val1 == val2) {
                throw std::runtime_error("Valores duplicados em x.");
            }
        }
    }
    double result = 0.0;

    for (size_t j = 0; j < n; ++j) {
        const T& xj = x[j];
        const T& yj = y[j];

        double term = static_cast<double>(yj);

        for (size_t i = 0; i < n; ++i) {
            if (i != j) {
                const T& xi = x[i];

                // L_ij(x) = (x - x_i) / (x_j - x_i)
                double numerator = a - xi;
                double denominator = xj - xi; 
                term *= (numerator / denominator);
            }
        }
        result += term;
    }
    return result;
}

/*
    Para plots, precisamos usar Baricentro que calcula os pesos uma vez e torna
    a checagem de x de O(n^2) para O(n).
*/

template <typename T>
class LagrageInterpBari {
private:
    std::vector<T> x_nodes;
    std::vector<T> y_nodes;
    std::vector<double> weights; 

public:
    LagrageInterpBari(const std::vector<T>& x, const std::vector<T>& y)
        : x_nodes(x), y_nodes(y) {

        if (x.size() != y.size()) throw std::invalid_argument("Size mismatch");

        size_t n = x.size();
        weights.resize(n);

        // 1. Pre calcular os pesos O(N^2)
        for (size_t j = 0; j < n; ++j) {
            double w = 1.0;
            for (size_t i = 0; i < n; ++i) {
                if (i != j) {
                    if (x[j] == x[i]) throw std::runtime_error("Duplicate X found");
                    w *= (x[j] - x[i]);
                }
            }
            weights[j] = 1.0 / w;
        }
    }

    double evaluate(double a) const {
        double numerator = 0.0;
        double denominator = 0.0;

        for (size_t j = 0; j < x_nodes.size(); ++j) {
            /*
                Usando \varepsilon pequeno para evitar divisão por zero.
            */
            if (std::abs(a - x_nodes[j]) < 1e-9) {
                return y_nodes[j];
            }
            double temp = weights[j] / (a - x_nodes[j]);
            numerator += temp * y_nodes[j];
            denominator += temp;
        }

        return numerator / denominator;
    }
};

int main()
{
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


    std::cout <<"Resultados 1ª func:\n" << result << "\n\nResultados 2ª func:\n" << resultBari << "\n";
    return 0;
}