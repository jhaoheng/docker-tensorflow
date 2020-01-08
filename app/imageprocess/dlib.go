package imageprocess

/*
目的在判斷一張圖中, 是否有大頭
若有大頭則截圖, 且大頭需滿足條件
1. 大小
2. 是否有誤判
*/

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"

	"github.com/Kagami/go-face"
	"os"
	"strings"
)

/*
The model only support JPG format
*/
type DLIBCLIENT struct {
	rec *face.Recognizer

	Pic PICTURE
}

type PICTURE struct {
	Img       image.Image // 原始圖片
	Byte      []byte      // 原始圖片 byte
	B64Pic    string
	Extension string      // 副檔名
	Faces     []face.Face // 辨識出來的 face 物件
	CropFaces []CROPFACE  // 切出來的大頭
}

type CROPFACE struct {
	Vector []float32
	Img    image.Image
	Byte   []byte
	B64Pic string // base64
	Bounds image.Rectangle
	Size   image.Point
}

func NEWDlibClient() (DLIBCLIENT, error) {
	dlibClient := DLIBCLIENT{}
	rec, err := face.NewRecognizer(dlibClient.GetModelsPath())
	if err == nil {
		dlibClient.rec = rec
	}
	return dlibClient, nil
}

// Free the resources when you're finished.
func (dlibClient *DLIBCLIENT) Close() {
	if dlibClient.rec != nil {
		dlibClient.rec.Close()
	}
}

func (dlibClient *DLIBCLIENT) GetModelsPath() (modelPath string) {
	pwd, _ := os.Getwd()
	if strings.Contains(pwd, "imageprocess") {
		modelPath = "./dlibmodels"
	} else {
		modelPath = "./imageprocess/dlibmodels"
	}
	return
}

func (dlibClient *DLIBCLIENT) SetJPG_And_ProcessIt(byteJPG []byte) (crop_faces []CROPFACE, err error) {
	defer func() {
		if err := recover(); err != nil {
			fmt.Println(err)
			return
		}
	}()

	//
	dlibClient.SetJPG(byteJPG)
	dlibClient.ExtractFaces()
	dlibClient.CropFaces()
	crop_faces = dlibClient.Pic.CropFaces
	return
}

/*
if Decode Fail, err != nil
*/
func (dlibClient *DLIBCLIENT) SetJPG(byteJPG []byte) {
	dlibClient.Pic = PICTURE{}
	img, extension, err := image.Decode(bytes.NewReader(byteJPG))
	if err != nil {
		panic(fmt.Sprintf("SetJPG : %v", err))
	}
	dlibClient.Pic.Img = img
	dlibClient.Pic.Extension = extension
	dlibClient.Pic.Byte = byteJPG
	dlibClient.Pic.B64Pic = base64.StdEncoding.EncodeToString(byteJPG)
	return
}

func (dlibClient *DLIBCLIENT) ExtractFaces() {
	faces, err := dlibClient.rec.Recognize(dlibClient.Pic.Byte)
	if err != nil {
		panic(fmt.Sprintf("ExtractFaces : %v", err))
	}
	dlibClient.Pic.Faces = faces
	return
}

func (dlibClient *DLIBCLIENT) CropFaces() {
	dlibClient.Pic.CropFaces = []CROPFACE{}
	for _, f := range dlibClient.Pic.Faces {
		crop_feature := dlibClient.crop(f)
		dlibClient.Pic.CropFaces = append(dlibClient.Pic.CropFaces, crop_feature)
	}
}

func (dlibClient *DLIBCLIENT) crop(f face.Face) (crop_face CROPFACE) {
	img := dlibClient.Pic.Img

	type subImager interface {
		SubImage(image.Rectangle) image.Image
	}
	var cropImg image.Image
	if si, ok := img.(subImager); ok {
		cropImg = si.SubImage(f.Rectangle)
	}

	var b bytes.Buffer
	w := bufio.NewWriter(&b)
	err := jpeg.Encode(w, cropImg, &jpeg.Options{Quality: 100})
	if err != nil {
		panic(fmt.Sprintf("Crop : %v", err))
	}

	/*
		Need to add age & gender
	*/
	crop_face = CROPFACE{
		Vector: f.Descriptor[:],
		Img:    cropImg,
		Byte:   b.Bytes(),
		B64Pic: base64.StdEncoding.EncodeToString(b.Bytes()),
		Bounds: cropImg.Bounds(),
		Size:   cropImg.Bounds().Size(),
	}
	return
}

/*
This method is for test
*/
func (dlibClient *DLIBCLIENT) ReadJPGFromPath(jpgPath string) []byte {
	b, err := ioutil.ReadFile(jpgPath)
	if err != nil {
		panic(err)
	}
	return b
}
