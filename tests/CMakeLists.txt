add_executable(Neural_Layer_Test Neural_Layer_Test.cpp)
target_link_libraries(Neural_Layer_Test CppNNet)
add_test(NAME FeedForwardTest COMMAND Neural_Layer_Test)

add_executable(Normalizer_Test Normalizer_Test.cpp)
target_link_libraries(Normalizer_Test CppNNet)
add_test(NAME NormalizerTest COMMAND Normalizer_Test ${CMAKE_SOURCE_DIR}/training_data/simpledata_set.csv)

add_executable(Neural_Training_Test Neural_Training_Test.cpp)
target_link_libraries(Neural_Training_Test CppNNet)
add_test(NAME TrainingTest COMMAND Neural_Training_Test ${CMAKE_SOURCE_DIR}/training_data/simpledata_set.csv)

add_executable(Iris_Dataset_Test Iris_Dataset_Test.cpp)
target_link_libraries(Iris_Dataset_Test CppNNet)
add_test(NAME IrisDatasetTest COMMAND Iris_Dataset_Test ${CMAKE_SOURCE_DIR}/training_data/iris_dataset.csv)

add_executable(MAGIC_Net_Test MAGIC_Net_Test.cpp)
target_link_libraries(MAGIC_Net_Test CppNNet)
add_test(NAME MAGICNetTest COMMAND MAGIC_Net_Test)

add_executable(MSGD_Neural_Training_Test MSGD_Neural_Training_Test.cpp)
target_link_libraries(MSGD_Neural_Training_Test CppNNet)
add_test(NAME MSGD_TrainingTest COMMAND MSGD_Neural_Training_Test ${CMAKE_SOURCE_DIR}/training_data/simpledata_set.csv)

add_executable(MSGD_Iris_Dataset_Test MSGD_Iris_Dataset_Test.cpp)
target_link_libraries(MSGD_Iris_Dataset_Test CppNNet)
add_test(NAME MSGD_IrisDatasetTest COMMAND MSGD_Iris_Dataset_Test ${CMAKE_SOURCE_DIR}/training_data/iris_dataset.csv)

add_executable(MSGD_MAGIC_Net_Test MSGD_MAGIC_Net_Test.cpp)
target_link_libraries(MSGD_MAGIC_Net_Test CppNNet)
add_test(NAME MSGD_MAGICNetTest COMMAND MSGD_MAGIC_Net_Test)

add_executable(Iris_Dataset_gz_Test Iris_Dataset_gz_Test.cpp)
target_link_libraries(Iris_Dataset_gz_Test CppNNet)
add_test(NAME IrisDatasetGZTest COMMAND Iris_Dataset_gz_Test ${CMAKE_SOURCE_DIR}/training_data/iris_dataset.csv.gz)
