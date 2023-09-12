// Copyright 2023 The Ent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"sort"

	"github.com/bugra/kmeans"
	"github.com/pointlander/datum/iris"
)

// MNIST cluster the MNIST data set
func MNIST() {
}

// SelfAware is the self aware mode
func SelfAware(datum iris.Datum, fisher Matrix) {

}

var (
	// MNIST MNIST mode
	FlagMNIST = flag.Bool("mnist", false, "mnist mode")
)

func main() {
	flag.Parse()

	if *FlagMNIST {
		MNIST()
		return
	}

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	fisher := NewMatrix(0, 4, len(datum.Fisher))
	for _, embedding := range datum.Fisher {
		fisher.Data = append(fisher.Data, embedding.Measures...)
	}

	units := Normalize(fisher)
	projected := PCA(units)
	embedded := SelfAttention(projected, projected, projected)
	type Vector struct {
		Index int
		Value []float64
		Label string
	}
	vectors := make([]Vector, 0, len(embedded.Data))
	for i := 0; i < embedded.Rows; i++ {
		v := embedded.Data[i*embedded.Cols : i*embedded.Cols+embedded.Cols]
		vectors = append(vectors, Vector{
			Index: i,
			Value: v,
			Label: datum.Fisher[i].Label,
		})
	}
	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Value[0] < vectors[j].Value[0]
	})
	for _, v := range vectors {
		fmt.Printf("%s %f\n", datum.Fisher[v.Index].Label, v.Value)
	}
	rawData := make([][]float64, len(vectors))
	for i, v := range vectors {
		rawData[i] = v.Value
	}
	clusters, err := kmeans.Kmeans(rawData, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Println(vectors[i].Label, i, v)
	}
}
