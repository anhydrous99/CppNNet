add_library(CppNNet CSV_Importer.cpp Normalizer.cpp Neural_Layer.cpp Neural_Trainer.cpp MSGD_Neural_Trainer.cpp)
target_link_libraries(CppNNet OpenMP::OpenMP_CXX Eigen3::Eigen CURL::libcurl)
