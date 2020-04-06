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
	- dlib could output the image base64 format, you could use `https://codebeautify.org/base64-to-image-converter` to show the picture.


# 註記, tf2 因應模型的不同, 需要知道的變數如下 : 
1. 是否圖片要先預處理
	- constructGraphToNormalizeImage() : `(value - Mean)/Scale`
2. 模型資料夾路徑
3. 模型名稱 : 不等於模型資料夾名稱
	- 可用 `saved_model_cli` 判斷
4. 得知 feeds:output_name 與 fetchs:output_name
	- 可用 `saved_model_cli` 判斷
