#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// https://github.com/nlohmann/json/
using json = nlohmann::json;

namespace data_loader
{

struct element
{
    double x;
    double y;
    double r;
    double r2;

    double scale;
    std::string filename;

    glm::vec3 position()
    {
        return glm::vec3(x, -y, 0);
    }
};


void to_json(json& j, const element& d)
{
    j = json{{"x", d.x}, {"y", d.y}, {"r", d.r}, {"r2", d.r2},
        {"scale", d.scale}, {"filename", d.filename}};
}   

void from_json(const json& j, element& d)
{
    j.at("x").get_to(d.x);
    j.at("y").get_to(d.y);
    j.at("r").get_to(d.r);
    j.at("r2").get_to(d.r2);
    j.at("scale").get_to(d.scale);
    j.at("filename").get_to(d.filename);
}

std::vector<data_loader::element> load_json(const std::string filename)
{
    std::ifstream m_file(filename);
    json j;
    m_file >> j;

    std::vector<data_loader::element> elements;
    for (auto& element : j) {
        elements.push_back(element.get<data_loader::element>());
    }

    return elements;
}

}