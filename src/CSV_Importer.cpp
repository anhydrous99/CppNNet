#include "CSV_Importer.h"

#include "curl/curl.h"
#include <iostream>
#include <fstream>
#include <random>
#include <zlib.h>

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

#define CHUNK 16384

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets) {
  _filename = filename;
  _sof = size_of_samples;
  _sot = size_of_targets;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, bool reverse) :
    CSV_Importer(filename, size_of_samples, size_of_targets) {
  _reverse = reverse;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter) :
    CSV_Importer(filename, size_of_samples, size_of_targets) {
  _delim = delimiter;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, char delimiter,
                                    bool reverse) : CSV_Importer(filename, size_of_samples, size_of_targets,
                                                                 delimiter) {
  _reverse = reverse;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx) {
  _start_idx = start_idx;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx,
                                    bool reverse) : CSV_Importer(filename, size_of_samples, size_of_targets,
                                                                 start_idx) {
  _reverse = reverse;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx,
                                    char delimiter) :
    CSV_Importer(filename, size_of_samples, size_of_targets, start_idx) {
  _delim = delimiter;
}

CppNNet::CSV_Importer::CSV_Importer(std::string &filename, int size_of_samples, int size_of_targets, int start_idx,
                                    char delimiter, bool reverse) : CSV_Importer(filename, size_of_samples,
                                                                                 size_of_targets,
                                                                                 start_idx, delimiter) {
  _reverse = reverse;
}

CppNNet::CSV_Importer::~CSV_Importer() {
  if (_curlinit)
    curl_global_cleanup();
}

void CppNNet::CSV_Importer::ObtainData() {
  int len = _filename.length();
  bool gotten = false;
  MemoryStruct mem;
  if (len > 4) {
    if (_filename.substr(0, 4) == "http") {
      std::cout << " Downloading File: " << _filename << std::endl;
      mem = Binary_Downloader();
      gotten = true;
    }
  }

  if (!gotten) {
    std::ifstream fin(_filename, std::ios::in | std::ios::binary);
    if (fin) {
      fin.seekg(0, fin.end);
      mem.size = static_cast<size_t>(fin.tellg());
      fin.seekg(0, fin.beg);
      mem.memory = (unsigned char *) malloc(mem.size * sizeof(unsigned char));

      std::cout << " Reading File: " << _filename << std::endl;
      // read data as a block
      fin.read((char *) mem.memory, mem.size);

      if (!fin) {
        std::cerr << "Error: only " << fin.gcount() << " could be read";
        delete[] mem.memory;
        fin.close();
        return;
      }
      gotten = true;
      fin.close();
    }
  }

  if (len > 8) {
    if (_filename.substr(len - 7) == ".csv.gz") {
      MemoryStruct to;
      to.memory = (unsigned char *) malloc(1);
      to.size = 0;

      int ret = doinflate(&mem, &to);
      if (ret != Z_OK) {
        zerr(ret);
        return;
      }

      // move to to mem
      delete[] mem.memory;
      mem.memory = to.memory;
      mem.size = to.size;
    }
  }

  std::string buffer((char *) mem.memory, mem.size);
  parse(buffer);
  _hasdata = true;
  delete[] mem.memory;
}

/* call back for binary files */
static size_t Binary_WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  size_t realsize = size * nmemb;
  auto *mem = (CppNNet::MemoryStruct *) userp;

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

CppNNet::MemoryStruct CppNNet::CSV_Importer::Binary_Downloader() {
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
  return chunk;
}

int CppNNet::CSV_Importer::doinflate(const MemoryStruct *src, MemoryStruct *dst) {
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
    memcpy(in, (const void *) (src->memory + i * CHUNK), CHUNK);
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
  } while (ret != Z_STREAM_END);

  (void) inflateEnd(&strm);
  return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

void CppNNet::CSV_Importer::parse(std::string &to_parse) {
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

void CppNNet::CSV_Importer::zerr(int ret) {
  std::cerr << "CSV_Importer: ";
  switch (ret) {
    case Z_ERRNO:
      std::cerr << "Error reading from mem:\n";
      break;
    case Z_STREAM_ERROR:
      std::cerr << "Invalid compression level\n";
      break;
    case Z_DATA_ERROR:
      std::cerr << "Invalid or incomplete deflate data\n";
      break;
    case Z_MEM_ERROR:
      std::cerr << "out of memory\n";
      break;
    case Z_VERSION_ERROR:
      std::cerr << "zlib version mismatch\n";
      break;
    default:
      std::cerr << "Unknown Error\n";
  }
}

std::vector<std::string> CppNNet::CSV_Importer::GetData() {
  if (_hasdata)
    return _data;
  else {
    ObtainData();
    return _data;
  }
}

std::vector<Eigen::VectorXf> CppNNet::CSV_Importer::GetSamples() {
  if (!_hasdata)
    ObtainData();

  std::vector<Eigen::VectorXf> output;
  int s = _sof + _sot;
  if (!_reverse) {
    for (unsigned long i = 0, sizei = _data.size() / s; i < sizei; i++) {
      Eigen::VectorXf nm(_sof);
      for (int j = 0, sizej = _sof; j < sizej; j++)
        nm[j] = std::stof(_data[i * s + j]);
      output.push_back(nm);
    }
  } else {
    for (unsigned long i = 0, sizei = _data.size() / s; i < sizei; i++) {
      Eigen::VectorXf nm(_sof);
      for (int j = _sof; j < s; j++)
        nm[j - _sof] = std::stof(_data[i * s + j]);
      output.push_back(nm);
    }
  }
  return output;
}

std::vector<Eigen::VectorXf> CppNNet::CSV_Importer::GetTargets() {
  if (!_hasdata)
    ObtainData();

  std::vector<Eigen::VectorXf> output;
  int s = _sof + _sot;
  if (!_reverse) {
    for (unsigned long i = 0, sizei = _data.size() / s; i < sizei; i++) {
      Eigen::VectorXf nm(_sot);
      for (int j = _sof; j < s; j++)
        nm[j - _sof] = std::stof(_data[i * s + j]);
      output.push_back(nm);
    }
  } else {
    for (unsigned long i = 0, sizei = _data.size() / s; i < sizei; i++) {
      Eigen::VectorXf nm(_sof);
      for (int j = 0, sizej = _sof; j < sizej; j++)
        nm[j] = std::stof(_data[i * s + j]);
      output.push_back(nm);
    }
  }
  return output;
}

CppNNet::DataSet CppNNet::CSV_Importer::GetDataSet() {
  if (!_hasdata)
    ObtainData();

  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_int_distribution<unsigned long> dist;

  std::vector<Eigen::VectorXf> samples = GetSamples();
  std::vector<Eigen::VectorXf> targets = GetTargets();

  unsigned long n = samples.size();
  unsigned long k = n / _val;
  std::vector<Eigen::VectorXf> samples_validation(samples.begin(), samples.begin() + k);
  std::vector<Eigen::VectorXf> targets_validation(targets.begin(), targets.begin() + k);

  for (unsigned long i = k; i < n; i++) {
    unsigned long dd = dist(g, std::uniform_int_distribution<unsigned long>::param_type(0, i));
    if (dd < k) {
      std::swap(samples_validation[dd], samples[i]);
      std::swap(targets_validation[dd], targets[i]);
    }
  }

  DataSet dataSet;
  dataSet.sample_validation_set = std::move(samples_validation);
  dataSet.target_validation_set = std::move(targets_validation);
  dataSet.sample_training_set = std::vector<Eigen::VectorXf>(samples.begin() + k + 1, samples.end());
  dataSet.target_training_set = std::vector<Eigen::VectorXf>(targets.begin() + k + 1, targets.end());
  return dataSet;
}