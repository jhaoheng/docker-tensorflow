package imageprocess

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type TFPCLIENT struct {
	ModelGraph   *tf.Graph
	ModelSession *tf.Session
}

func NewTFClient(modelfile string) TFPCLIENT {
	// load model
	pwd, _ := os.Getwd()
	if strings.Contains(pwd, "imageprocess") {
		modelfile = "./tfmodels/" + modelfile
	} else {
		modelfile = "./imageprocess/tfmodels/" + modelfile
	}
	model, err := ioutil.ReadFile(modelfile)
	if err != nil {
		panic(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		panic(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}

	tfClient := TFPCLIENT{
		ModelGraph:   graph,
		ModelSession: session,
	}
	return tfClient
}

/*
Close session
*/
func (tfClient *TFPCLIENT) Close() {
	tfClient.ModelSession.Close()
}

/*
image vector 轉換成 tenfor format
*/
func (tfClient *TFPCLIENT) MakeTensorFromImageVector(vector [][]float32) (tensor *tf.Tensor) {
	tensor, err := tf.NewTensor(vector)
	if err != nil {
		panic(err)
	}
	return
}

/*
image byte 轉換成 tensor format
*/
func (tfClient *TFPCLIENT) MakeTensorFromImageByte(imgBytes []byte) *tf.Tensor {
	tensor, err := tf.NewTensor(string(imgBytes))
	if err != nil {
		panic(err)
	}
	// Construct a graph to normalize the image
	graph, input, output, err := tfClient.constructGraphToNormalizeImage()
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

/*
取得 Result
*/
func (tfClient *TFPCLIENT) GetResult(feedsOutput map[tf.Output]*tf.Tensor, fetchOutput []tf.Output) (output []*tf.Tensor) {
	output, err := tfClient.ModelSession.Run(
		feedsOutput,
		fetchOutput,
		nil)
	if err != nil {
		panic(err)
	}
	return output
}

/*
顯示 model input/output 可用的參數名稱
*/
func (tfClient *TFPCLIENT) ShowGraphOperation(graph *tf.Graph) {
	for i, obj := range graph.Operations() {
		fmt.Println(i, "=>", obj.Name(), ",", obj.Type(), ",", obj.Output(0).Shape(), ",", obj.NumOutputs())
	}
}

/*
正規化設定
*/
func (tfClient *TFPCLIENT) constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 64, 64
		Mean  = float32(0)
		Scale = float32(255)
	)

	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	// fmt.Println("input =>", input.Op.Name())
	// stop()
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()

	return graph, input, output, err
}

/*
This method is for test
*/
func (tfClient *TFPCLIENT) ReadJPGFromPath(jpgPath string) []byte {
	b, err := ioutil.ReadFile(jpgPath)
	if err != nil {
		panic(err)
	}
	return b
}
