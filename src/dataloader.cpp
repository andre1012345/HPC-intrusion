#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<DataRow> loadCSV(const std::string& filename) {
    std::vector<DataRow> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line, word;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        DataRow row;
        
        // In our cleaned CSV, the last column is the Label (0 or 1)
        // and everything before it is a Feature.
        std::vector<double> values;
        while (std::getline(ss, word, ',')) {
            values.push_back(std::stod(word));
        }

        if (!values.empty()) {
            // Last value is the label, the rest are features
            row.label = std::to_string((int)values.back()); 
            values.pop_back();
            row.features = values;
            data.push_back(row);
        }
    }

    file.close();
    return data;
}