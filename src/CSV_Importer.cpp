#include "CSV_Importer.h"

#include "curl/curl.h"
#include <iostream>
#include <fstream>
#include <zlib.h>

#define CHUNK 16384

CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets) {
  _filename = filename;
  _sof = size_of_samples;
  _sot = size_of_targets;
}

CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter) :
    CSV_Importer(filename, size_of_samples, size_of_targets) {
  _delim = delimiter;
}

CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx) {
  _start_idx = start_idx;
}

CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx,
                           char delimiter) :
    CSV_Importer(filename, size_of_samples, size_of_targets, start_idx) {
  _delim = delimiter;
}

CSV_Importer::~CSV_Importer() {
  if (_curlinit)
    curl_global_cleanup();
}

void CSV_Importer::ObtainData() {
  int len = _filename.length();
  if (len > 3) {
    if (_filename.substr(len - 7 - 1, 7) == ".csv.gz") {
      // TODO if file is compressed, download, decompress it, and add to _data
    }
  }

  if (len > 4) {
    if (_filename.substr(0, 4) == "http") {
      std::string input_string = String_Downloader();
      parse(input_string);
      _hasdata = true;
      return;
    }
  }

  std::ifstream file(_filename);
  std::string tmp;
  std::string line;
  if (file.is_open()) {
    int i = 0;
    while (getline(file, line)) {
      i++;
      if (i - 1 < _start_idx) continue;

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
static size_t String_WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((std::string *) userp)->append((char *) contents, size * nmemb);
  return size * nmemb;
}

/* call back for binary files */
static size_t Binary_WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  size_t realsize = size * nmemb;
  auto *mem = (MemoryStruct *) userp;

  auto *ptr = (unsigned char *) realloc(mem->memory, mem->size + realsize + 1);
  if (ptr == nullptr) {
    /* out of memory! */
    std::cerr << "not enough memory (realloc returned NULL)\n";
    return 0;
  }

  mem->memory = ptr;
  memcpy(&(mem->memory[mem->size]), contents, realsize);
  mem->size += realsize;
  mem->memory[mem->size] = 0;

  return realsize;
}

/* Used by CSV_Importer to display a progress bar */
// Progress bar
struct myprogress {
  double lastruntime;
  CURL *curl;
};

static int progress_func(void *ptr, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
  if ((float) dltotal <= 0.0) return 0;
  auto progress = static_cast<float>(dlnow) / static_cast<float>(dltotal);
  int barWidth = 70;
  int pos = static_cast<int>(barWidth * progress);
  std::cout << "[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "]" << int(progress * 100.0) << "%\r";
  std::cout.flush();
  return 0;
}

static int older_progress(void *p, double dltotal, double dlnow, double ultotal, double ulnow) {
  return progress_func(p, static_cast<curl_off_t>(dltotal), static_cast<curl_off_t>(dlnow),
                       static_cast<curl_off_t>(ultotal), static_cast<curl_off_t>(ulnow));
}

std::string CSV_Importer::String_Downloader() {
  CURL *curl;
  CURLcode res;
  std::string readBuffer;
  struct myprogress prog;
  curl = curl_easy_init();
  if (curl) {
    _curlinit = true;
    curl_easy_setopt(curl, CURLOPT_URL, _filename.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl/1.0");
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, String_WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    // for older libcurls
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, older_progress);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &prog);
    // for newer libcurls
#if LIBCURL_VERSION_NUM >= 0x072000
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_func);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
#endif
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    res = curl_easy_perform(curl);
    /* Check for errors */
    if (res != CURLE_OK)
      std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  std::cout << std::endl;
  return readBuffer;
}

MemoryStruct CSV_Importer::Binary_Downloader() {
  CURL *curl;
  CURLcode res;
  MemoryStruct chunk;

  chunk.memory = (unsigned char *) malloc(1);
  chunk.size = 0;

  struct myprogress prog;
  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, _filename.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl/1.0");
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, Binary_WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &chunk);
    // for older libcurls
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, older_progress);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &prog);
    // for newer libcurls
#if LIBCURL_VERSION_NUM >= 0x072000
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_func);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
#endif
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    res = curl_easy_perform(curl);
    /* Check for errors */
    if (res != CURLE_OK)
      std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  std::cout << std::endl;
}

int CSV_Importer::doinflate(const MemoryStruct *src, MemoryStruct *dst) {
  int ret, flush;
  unsigned have;
  z_stream strm;
  unsigned char in[CHUNK];
  unsigned char out[CHUNK];

  /* allocate inflate state */
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  strm.avail_in = 0;
  strm.next_in = Z_NULL;
  ret = inflateInit2(&strm, (15 + 32));
  if (ret != Z_OK)
    return ret;

  /* decompress until deflate stream ends or end of length */
  int i = 0;
  do {
    std::memcpy(in, (const void *) (src->memory[i * CHUNK]), CHUNK);
    strm.avail_in = CHUNK;
    if (strm.avail_in == 0)
      break;
    strm.next_in = in;

    /* run inflate() on input until output buffer not full */
    do {
      strm.avail_out = CHUNK;
      strm.next_out = out;
      ret = inflate(&strm, Z_NO_FLUSH);
      assert(ret != Z_STREAM_ERROR); /* state not clobbered */
      switch (ret) {
        case Z_NEED_DICT:
          ret = Z_DATA_ERROR;     /* and fall through */
        case Z_DATA_ERROR:
        case Z_MEM_ERROR:
          (void) inflateEnd(&strm);
          return ret;
        default:
          break;
      }
      have = CHUNK - strm.avail_out;

      // Reallocate memory
      MemoryStruct *mem = dst;
      auto *ptr = (unsigned char *) realloc(mem->memory, mem->size + have);
      if (ptr == nullptr) {
        /* out of memory! */
        std::cerr << "not enought memory (realloc returned NULL)\n";
        (void) inflateEnd(&strm);
        return Z_ERRNO;
      }

      // Copy out to the newly allocated memory
      mem->memory = ptr;
      memcpy(&(mem->memory[mem->size]), out, have);
      mem->size += have;
      mem->memory[mem->size] = 0;

    } while (strm.avail_out == 0);
    i++;
    /* done when inflate() says it's done */
  } while (strm.avail_out != Z_STREAM_END);

  (void) inflateEnd(&strm);
  return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

void CSV_Importer::parse(std::string &to_parse) {
  std::stringstream stream1(to_parse);
  std::string line, tmp;
  int i = 0;
  while (getline(stream1, line)) {
    i++;
    if (i - 1 < _start_idx) continue;

    std::stringstream ss(line);
    while (std::getline(ss, tmp, _delim)) {
      _data.push_back(tmp);
    }
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