package main

import (
	"fmt"
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	fmt.Printf("\n== Tensorflow CoreLib Version is : %v ==\n\n", tf.Version())

	//
	image := "/go/src/app/testimg/demo.jpg"
	modelDir := "/go/src/app/tfmodel2/age_gender_v2"
	modelName := "serve"
	feedOutputName := "serving_default_input_1"
	fetchOutputName := "StatefulPartitionedCall"

	//
	b, err := ioutil.ReadFile(image)
	if err != nil {
		panic(err)
	}
	tensor := MakeTensorFromImageByte(b)

	dir := modelDir
	tags := []string{
		modelName,
	}

	model, err := tf.LoadSavedModel(dir, tags, nil)
	if err != nil {
		panic(err)
	}

	for i, obj := range model.Graph.Operations() {
		fmt.Printf("%v => %v, %v, %v\n", i, obj.Name(), obj.Type(), obj.NumOutputs())
	}

	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation(feedOutputName).Output(0): tensor,
		},
		[]tf.Output{
			model.Graph.Operation(fetchOutputName).Output(0),
			model.Graph.Operation(fetchOutputName).Output(1),
		},
		nil,
	)
	fmt.Println(result[0].Value().([][]float32)[0][0]*35 + 35)
	if result[1].Value().([][]float32)[0][0] > result[1].Value().([][]float32)[0][1] {
		fmt.Println("women")
	} else {
		fmt.Println("man")
	}
}

func MakeTensorFromImageByte(imgBytes []byte) *tf.Tensor {
	tensor, err := tf.NewTensor(string(imgBytes))
	if err != nil {
		panic(err)
	}
	// Construct a graph to normalize the image
	graph, input, output, err := constructGraphToNormalizeImage()
	if err != nil {
		panic(err)
	}
	// Execute that graph to normalize this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		panic(err)
	}
	return normalized[0]
}

func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	var TFStringType tf.DataType = tf.String
	var TFFloatType tf.DataType = tf.Float
	const (
		H, W  = 128, 128
		Mean  = float32(127.5)
		Scale = float32(127.5)
	)

	s := op.NewScope()
	input = op.Placeholder(s, TFStringType)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), TFFloatType),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()

	return graph, input, output, err
}
