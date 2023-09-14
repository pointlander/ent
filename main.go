// Copyright 2023 The Ent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/kmeans"
)

// MNIST cluster the MNIST data set
func MNIST() {
}

// Sample samples the embedding function for the iris data set
func Sample() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	sample := func(rngSeed int64) (x [150][]int) {
		fisher := NewMatrix(0, 4, len(datum.Fisher))
		for _, embedding := range datum.Fisher {
			for _, measure := range embedding.Measures {
				fisher.Data = append(fisher.Data /*math.Abs(measure+rnd.NormFloat64()*0.1)*/, measure)
			}
		}

		units := Normalize(fisher)
		projected := PCA(units)
		embedded := SelfAttention(projected, projected, projected)

		type Vector struct {
			Index int
			Value []float64
		}
		vectors := make([]Vector, 0, len(embedded.Data))
		for i := 0; i < embedded.Rows; i++ {
			v := embedded.Data[i*embedded.Cols : i*embedded.Cols+embedded.Cols]
			vectors = append(vectors, Vector{
				Index: i,
				Value: v,
			})
		}

		rawData := make([][]float64, len(vectors))
		for i, v := range vectors {
			rawData[i] = v.Value
		}
		clusters, _, err := kmeans.Kmeans(rngSeed, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := range x {
			x[i] = make([]int, 150)
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					x[i][j]++
				}
			}
		}
		return x
	}

	var sum [150][]int
	for i := range sum {
		sum[i] = make([]int, 150)
	}
	for i := 0; i < 100; i++ {
		x := sample(int64(i) + 1)
		for i := range sum {
			for j := range sum[i] {
				sum[i][j] += x[i][j]
			}
		}
	}

	fisher := NewMatrix(0, 150, len(datum.Fisher))
	for _, embedding := range sum {
		for _, measure := range embedding {
			fisher.Data = append(fisher.Data, float64(measure))
		}
	}

	units := Normalize(fisher)
	projected := PCA(units)
	embedded := SelfAttention(projected, projected, projected)

	type Vector struct {
		Index int
		Value []float64
	}
	vectors := make([]Vector, 0, len(embedded.Data))
	for i := 0; i < embedded.Rows; i++ {
		v := embedded.Data[i*embedded.Cols : i*embedded.Cols+embedded.Cols]
		vectors = append(vectors, Vector{
			Index: i,
			Value: v,
		})
	}

	rawData := make([][]float64, len(vectors))
	for i, v := range vectors {
		rawData[i] = v.Value
	}
	clusters, _, err := kmeans.Kmeans(1, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Println(datum.Fisher[i].Label, i, v)
	}
}

var (
	// MNIST MNIST mode
	FlagMNIST = flag.Bool("mnist", false, "mnist mode")
	// FlagSample sample mode
	FlagSample = flag.Bool("sample", false, "sample mode")
)

func main() {
	flag.Parse()

	if *FlagMNIST {
		MNIST()
		return
	}

	if *FlagSample {
		Sample()
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
	/*sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Value[0] < vectors[j].Value[0]
	})
	for _, v := range vectors {
		fmt.Printf("%s %f\n", datum.Fisher[v.Index].Label, v.Value)
	}*/
	rawData := make([][]float64, len(vectors))
	for i, v := range vectors {
		rawData[i] = v.Value
	}
	clusters, _, err := kmeans.Kmeans(3, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Println(vectors[i].Label, i, v)
	}
}
