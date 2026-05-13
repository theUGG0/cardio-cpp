
#include <cstddef>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include "../cardio.hpp"

using namespace std;

vector<float> read_csv_data(string filename){
    regex re{","};
    ifstream file{filename};
    vector<float> data{};
    string line{};
    
    //skip header
    getline(file, line);
    
    while (getline(file, line)) {
        vector tokens(sregex_token_iterator(line.begin(),line.end(), re, -1), {});
        data.push_back(stof(tokens[2]));
    }
    return data;
}

int main(){
    vector<float> data{read_csv_data("100_ekg.csv")};
    PanTompkins<360> h{};
    //h._pam_tompkins(data, 360);
    size_t idx{};
    for(auto x : data){
        h.push_data(x, ++idx);
    }
}