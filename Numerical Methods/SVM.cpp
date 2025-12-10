#include <vector>
#include <iostream>
#include "SVM.h"


void runSVM() {
    std::vector<std::vector<double>> X_train = {
        {1.0, 1.0},
        {1.5, 1.5},
        {2.0, 1.0},

        {5.0, 5.0},
        {5.5, 5.5},
        {6.0, 5.0}
    };

    std::vector<double> y_train = {
        -1.0, -1.0, -1.0,  
         1.0,  1.0,  1.0   
    };


    std::cout << "Initializing SVM..." << std::endl;

    SVM<double> model(10.0, 0.001);

    std::cout << "Training..." << std::endl;
    model.fit(X_train, y_train);
    std::cout << "Training Complete.\n" << std::endl;

    
    std::vector<std::vector<double>> test_points = {
        {1.2, 1.2}, 
        {5.2, 5.2}, 
        {0.0, 0.0}, 
        {10.0, 10.0}
    };

    std::cout << "--- Predictions ---" << std::endl;
    for (const auto& point : test_points) {
        double prediction = model.predict(point);

        std::cout << "Point (" << point[0] << ", " << point[1] << ") \t-> Predicted Class: " << prediction << std::endl;
    }

    std::cout << "\n--- Model Internals ---" << std::endl;
    std::cout << "Bias (b): " << model.get_bias() << std::endl;
}
