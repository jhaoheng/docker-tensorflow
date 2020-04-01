FROM golang:1.13

ENV GO111MODULE=on

RUN apt-get update
RUN apt-get install -y libnss3-dev libx11-dev vim

# dlib : https://github.com/Kagami/go-face
RUN apt-get install -y libdlib-dev libopenblas-dev libjpeg62-turbo-dev
RUN mkdir /usr/local/lib/pkgconfig/ && touch /usr/local/lib/pkgconfig/dlib-1.pc
RUN echo \
'libdir=/usr/lib/x86_64-linux-gnu\n\
includedir=/usr/include\n\
Name: dlib\n\
Description: Numerical and networking C++ library\n\
Version: 19.10.0\n\
Libs: -L${libdir} -ldlib -lblas -llapack\n\
Cflags: -I${includedir}\n\
Requires:' >> /usr/local/lib/pkgconfig/dlib-1.pc
RUN go get github.com/Kagami/go-face@01156987f993

# tensorflow
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz && \
    tar -C /usr/local -xzf *.tar.gz	&& rm *.tar.gz && ldconfig
RUN go get -v github.com/tensorflow/tensorflow@v2.1.0