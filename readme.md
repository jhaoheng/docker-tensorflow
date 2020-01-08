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