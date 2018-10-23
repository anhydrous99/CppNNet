#include "CSV_Importer.h"

#include "curl/curl.h"
#include <iostream>
#include <fstream>

CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets) {
  _filename = filename;
  _sof = size_of_samples;
  _sot = size_of_targets;
}

CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter) :
    CSV_Importer(filename, size_of_samples, size_of_targets) {
  _delim = delimiter;
}

void CSV_Importer::ObtainData() {
  std::ifstream file(_filename);
  std::string tmp;
  std::string line;
  if (file.is_open()) {
    while (getline(file, line)) {
      std::stringstream ss(line);
      while (std::getline(ss, tmp, _delim)) {
        _data.push_back(tmp);
      }
    }
    _hasdata = true;
    file.close();
  } else
    std::cout << "Unable to open file " << _filename << std::endl;
}

/* Used by CSV_Importer to put the web source into a string type */
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string *) userp)->append((char *) contents, size * nmemb);
  return size * nmemb;
}

std::string CSV_Importer::Downloader() {
  CURL *curl;
  CURLcode res;
  std::string readBuffer;
  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, _filename.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl/1.0");
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    /* Check for errors */
    if (res != CURLE_OK)
      std::cout << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
}

std::vector<std::string> CSV_Importer::GetData() {
  if (_hasdata)
    return _data;
  else {
    ObtainData();
    return _data;
  }
}

std::vector<Eigen::VectorXf> CSV_Importer::GetSamples() {
  if (!_hasdata)
    ObtainData();

  std::vector<Eigen::VectorXf> output;
  int s = _sof + _sot;
  for (unsigned long i = 0, sizei = _data.size() / s; i < sizei; i++) {
    Eigen::VectorXf nm(_sof);
    for (int j = 0, sizej = _sof; j < sizej; j++)
      nm[j] = std::stof(_data[i * s + j]);
    output.push_back(nm);
  }
  return output;
}

std::vector<Eigen::VectorXf> CSV_Importer::GetTargets() {
  if (!_hasdata)
    ObtainData();

  std::vector<Eigen::VectorXf> output;
  int s = _sof + _sot;
  for (unsigned long i = 0, sizei = _data.size() / s; i < sizei; i++) {
    Eigen::VectorXf nm(_sot);
    for (int j = _sof; j < s; j++)
      nm[j - _sof] = std::stof(_data[i * s + j]);
    output.push_back(nm);
  }
  return output;
}
