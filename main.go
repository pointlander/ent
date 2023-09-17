// Copyright 2023 The Ent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/kmeans"
)

// MNIST cluster the MNIST data set
func MNIST() {
}

// Single is single sample mode for the iris data set
func Single() {
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

// Sample samples the embedding function for the iris data set
func Sample() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	sample := func(rngSeed int64) (x [150][]int) {
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
	// FlagSingle single mode
	FlagSingle = flag.Bool("single", false, "single mode")
)

func main() {
	flag.Parse()

	if *FlagMNIST {
		MNIST()
		return
	}

	if *FlagSingle {
		Single()
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

	fisher := NewMatrix(0, len(datum.Fisher), len(datum.Fisher))
	for _, a := range datum.Fisher {
		for _, b := range datum.Fisher {
			sum := 0.0
			for i, value := range a.Measures {
				diff := value - b.Measures[i]
				sum += diff * diff
			}
			fisher.Data = append(fisher.Data, math.Sqrt(sum))
		}
	}

	units := Normalize(fisher)
	projected := units //PCA(units)
	embedding := SelfEntropy(projected, projected, projected)

	type Vector struct {
		Position []float64
		Value    float64
		Label    string
		G        []float64
	}
	vectors := make([]Vector, 0, len(embedding))
	for i, v := range embedding {
		vectors = append(vectors, Vector{
			Label:    datum.Fisher[i].Label,
			Position: datum.Fisher[i].Measures,
			Value:    -v,
		})
	}
	for i := range vectors {
		for j := range vectors {
			d := 0.0
			for i, value := range vectors[i].Position {
				diff := value - vectors[j].Position[i]
				d += diff * diff
			}
			vectors[i].G = append(vectors[i].G, d/(vectors[i].Value*vectors[j].Value))
		}
	}

	rawData := make([][]float64, len(vectors))
	for i, v := range vectors {
		rawData[i] = v.G
	}
	clusters, _, err := kmeans.Kmeans(4, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Println(vectors[i].Label, i, v)
	}
}
