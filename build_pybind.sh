

echo compiling

c++ -O3 -Wall -shared -std=c++14 -fPIC -fPIC -lpng -lz -lfreetype -lm\
    Wrapper_Code.cpp -o PyProp.so  \
    -L $(dirname `pwd`)/lib -lTUsine -lfparser\
    `python3 -m pybind11 --includes`\
    `root-config --cflags --glibs --ldflags`\
    -I $(dirname `pwd`)/include \
    -I pybind11/include
    

