add_executable(Neural_Layer_Test Neural_Layer_Test.cpp)
target_link_libraries(Neural_Layer_Test
                      CppNNet)
add_test (NAME FeedForwardTest COMMAND Neural_Layer_Test)
