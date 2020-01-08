package imageprocess

import (
	"fmt"
	"testing"
)

func Test_GetPicCrop(t *testing.T) {
	dlibProcess, err := NEWDlibClient()
	if err != nil {
		panic(err)
	}
	defer dlibProcess.Close()
	jpgPath := "/go/src/app/testimg/bona.jpg"
	b := dlibProcess.ReadJPGFromPath(jpgPath)
	dlibProcess.SetJPG(b)
	dlibProcess.ExtractFaces()
	dlibProcess.CropFaces()
	for _, imgObj := range dlibProcess.Pic.CropFaces {
		fmt.Println(imgObj.Vector)
	}
}

func Test_GetPicInfo(t *testing.T) {
	dlibProcess, err := NEWDlibClient()
	if err != nil {
		panic(err)
	}
	defer dlibProcess.Close()
	jpgPath := "/go/src/app/testimg/bona.jpg"
	b := dlibProcess.ReadJPGFromPath(jpgPath)
	dlibProcess.SetJPG(b)
	fmt.Println(dlibProcess.Pic.B64Pic)
}
