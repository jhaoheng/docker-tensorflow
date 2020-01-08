package imageprocess

import (
	"fmt"
	"strconv"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func Test_Byte(t *testing.T) {
	tfClient := NewTFClient("age_gender_andy.pb")
	defer tfClient.Close()
	b := tfClient.ReadJPGFromPath("/go/src/app/testimg/bona.jpg")
	imgTensorFormat := tfClient.MakeTensorFromImageByte(b)
	feedsOutput := map[tf.Output]*tf.Tensor{
		tfClient.ModelGraph.Operation("input_1").Output(0): imgTensorFormat,
	}
	fetchOutput := []tf.Output{
		tfClient.ModelGraph.Operation("gender/Softmax").Output(0),
		tfClient.ModelGraph.Operation("age/Sigmoid").Output(0),
	}

	output := tfClient.GetResult(feedsOutput, fetchOutput)

	// gender
	func(Obj []float32) {
		if Obj[0] > Obj[1] {
			fmt.Println("woman")
		} else {
			fmt.Println("man")
		}

	}(output[0].Value().([][]float32)[0])

	// age
	func(Obj []float32) {
		value, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", Obj[0]*70.0), 64)
		fmt.Println(value)
	}(output[1].Value().([][]float32)[0])
}
func Test_Vector(t *testing.T) {
	tfClient := NewTFClient("age_0506_1_paul.pb")
	defer tfClient.Close()
	vector := [][]float32{
		0: {
			-0.09448673, 0.00299672, -0.01515014, -0.00071408, -0.06696047,
			-0.0049022, -0.08832493, -0.08441716, 0.10042, -0.12468411,
			0.20260036, -0.04197515, -0.14975297, -0.07462036, -0.06249582,
			0.18168075, -0.17225666, -0.08203805, -0.00451902, 0.02619861,
			0.11311005, -0.00396955, -0.03501457, 0.06844106, -0.13450187,
			-0.33966419, -0.1363975, -0.07410653, 0.01615451, -0.04746429,
			-0.03059909, -0.00604952, -0.13447005, 0.00071517, 0.02433086,
			0.08668517, 0.00277232, -0.05964718, 0.17318156, 0.04651684,
			-0.20049778, 0.05507299, 0.02470755, 0.23595715, 0.19114207,
			0.10148107, 0.03782246, -0.12871203, 0.09433473, -0.18688148,
			0.02312719, 0.11872759, 0.06432167, 0.08553082, -0.03720454,
			-0.11119911, 0.0646906, 0.06421642, -0.14366747, 0.03704356,
			0.04679065, -0.07703813, 0.03313119, 0.01389773, 0.23284599,
			0.05472896, -0.12956738, -0.16850038, 0.11573525, -0.18238996,
			-0.09238973, 0.01605953, -0.1146273, -0.20592251, -0.31073043,
			0.04182778, 0.44439369, 0.16161928, -0.19643666, 0.07449412,
			-0.06447967, -0.03075972, 0.17345226, 0.16626933, 0.02761606,
			-0.05479746, -0.10204716, 0.00131448, 0.24683532, -0.05268429,
			-0.01487479, 0.20475397, -0.00205846, 0.0964495, 0.01042209,
			0.06929132, -0.09276011, 0.08609562, -0.01308689, -0.0279386,
			0.10474412, 0.01610085, 0.07669854, 0.12831081, -0.15205586,
			0.21715966, -0.03564852, 0.02630121, 0.08200177, 0.04180234,
			-0.13742532, -0.02357174, 0.13010965, -0.21079631, 0.21230085,
			0.17257966, 0.0704184, 0.12778719, 0.15773121, 0.12556361,
			-0.01399942, 0.00308351, -0.2362863, -0.02304427, 0.03029038,
			-0.02955271, 0.11292487, 0.01218744},
	}
	imgTensorFormat := tfClient.MakeTensorFromImageVector(vector)
	feedsOutput := map[tf.Output]*tf.Tensor{
		tfClient.ModelGraph.Operation("dense_4_input").Output(0): imgTensorFormat,
	}
	fetchOutput := []tf.Output{
		tfClient.ModelGraph.Operation("dense_6/Relu").Output(0),
	}
	output := tfClient.GetResult(feedsOutput, fetchOutput)

	fmt.Println("Result =>", output[0].Value().([][]float32)[0])
}
