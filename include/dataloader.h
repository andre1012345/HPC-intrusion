#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>

struct DataRow {
    std::vector<double> features;
    std::string label;
};

std::vector<DataRow> loadCSV(const std::string& filename);

#endif
