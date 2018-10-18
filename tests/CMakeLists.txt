add_executable(Neural_Layer_Test Neural_Layer_Test.cpp)
target_link_libraries(Neural_Layer_Test CppNNet)
add_test (NAME FeedForwardTest COMMAND Neural_Layer_Test)

add_executable(Normalizer_Test Normalizer_Test.cpp)
target_link_libraries(Normalizer_Test CppNNet)
add_test(NAME NormalizerTest COMMAND Normalizer_Test ${CMAKE_SOURCE_DIR}/training_data/simpledata_set.csv)

add_executable(Neural_Training_Test Neural_Training_Test.cpp)
target_link_libraries(Neural_Training_Test CppNNet)
add_test(NAME TrainingTest COMMAND Neural_Training_Test ${CMAKE_SOURCE_DIR}/training_data/simpledata_set.csv)
