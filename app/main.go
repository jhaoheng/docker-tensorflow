package main

import (
	"app/imageprocess"
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	fmt.Printf("\n== Tensorflow CoreLib Version is : %v ==\n\n", tf.Version())

	model_name := "age_gender_andy.pb"
	img_name := "bona.jpg"
	//
	feedsOutputOperationName := "input_1"
	fetchOutputOperationName := "gender/Softmax"

	RunJPG(model_name, img_name, feedsOutputOperationName, fetchOutputOperationName)
}

var ImgBasePath = "/go/src/app/testimg/"

func RunJPG(modelName string, imgName string, feedsOutputOperationName, fetchOutputOperationName string) {
	tfClient := imageprocess.NewTFClient(modelName)
	defer tfClient.Close()
	b := tfClient.ReadJPGFromPath(ImgBasePath + imgName)
	imgTensorFormat := tfClient.MakeTensorFromImageByte(b)
	feedsOutput := map[tf.Output]*tf.Tensor{
		// tfClient.ModelGraph.Operation("input_1").Output(0): imgTensorFormat,
		tfClient.ModelGraph.Operation(feedsOutputOperationName).Output(0): imgTensorFormat,
	}
	fetchOutput := []tf.Output{
		// tfClient.ModelGraph.Operation("age/Sigmoid").Output(0),
		tfClient.ModelGraph.Operation(fetchOutputOperationName).Output(0),
	}

	output := tfClient.GetResult(feedsOutput, fetchOutput)
	fmt.Println(output[0].Value())
}
