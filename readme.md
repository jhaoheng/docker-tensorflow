# Notice
1. This repo not contain tensorflow model. Please put your-model on the `app/imagepross/tfmodels`
2. How to use? please refer the `app/imageprocess/**_test.go`

# for test

1. `docker-compose up -d`
2. `docker exec -it docker-tensorflow /bin/bash`
3. `cd /go/src/app/imageprocess`
4. test tensorflow
	- `go test -run Test_Byte`
	- `go test -run Test_Vector`
5. test dlib
	- `test test -run Test_GetPicInfo`
	- `test test -run Test_GetPicCrop`