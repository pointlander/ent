// Copyright 2023 The Ent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"sort"

	"github.com/pointlander/datum/iris"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Meta is the meta data analysis mode
func Meta(datum iris.Datum) {
	adjacency := NewMatrix(0, len(datum.Fisher), len(datum.Fisher))
	for _, a := range datum.Fisher {
		for j := 0; j < len(datum.Fisher); j++ {
			b := datum.Fisher[j]
			matrix := NewMatrix(0, 4, 2)
			matrix.Data = append(matrix.Data, a.Measures...)
			matrix.Data = append(matrix.Data, b.Measures...)
			entropy := SelfEntropy(matrix, matrix, matrix)
			sum := 0.0
			for _, v := range entropy {
				sum += v
			}
			adjacency.Data = append(adjacency.Data, -sum)
		}
	}
	entropy := SelfEntropy(adjacency, adjacency, adjacency)
	type Entropy struct {
		Index int
		Value float64
	}
	entropyList := make([]Entropy, 0, len(entropy))
	for i, v := range entropy {
		entropyList = append(entropyList, Entropy{i, v})
	}
	sort.Slice(entropyList, func(i, j int) bool {
		return entropyList[i].Value < entropyList[j].Value
	})
	for _, v := range entropyList {
		fmt.Println(v.Index, v.Value)
	}
}

// Full process all of the data
func Full(datum iris.Datum) {
	matrix := NewMatrix(0, 4, len(datum.Fisher))
	for _, a := range datum.Fisher {
		matrix.Data = append(matrix.Data, a.Measures...)
	}
	entropy := SelfEntropy(matrix, matrix, matrix)
	type Entropy struct {
		Index int
		Value float64
	}
	entropyList := make([]Entropy, 0, len(entropy))
	for i, v := range entropy {
		entropyList = append(entropyList, Entropy{i, v})
	}
	sort.Slice(entropyList, func(i, j int) bool {
		return entropyList[i].Value < entropyList[j].Value
	})
	for _, v := range entropyList {
		fmt.Println(datum.Fisher[v.Index].Label, v.Value)
	}
}

var (
	// FlagMeta meta mode
	FlagMeta = flag.Bool("meta", false, "meta mode")
	// FlagFull full mode
	FlagFull = flag.Bool("full", false, "full mode")
)

func main() {
	flag.Parse()

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	values := make([]float64, 0, len(datum.Fisher)*4)
	for _, embedding := range datum.Fisher {
		values = append(values, embedding.Measures...)
	}
	data := mat.NewDense(len(datum.Fisher), 4, values)
	rows, cols := data.Dims()

	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		return
	}

	k := 4
	var projection mat.Dense
	var vector mat.Dense
	pc.VectorsTo(&vector)
	projection.Mul(data, vector.Slice(0, cols, 0, k))

	for i := 0; i < rows; i++ {
		for j := 0; j < k; j++ {
			datum.Fisher[i].Measures[j] = projection.At(i, j)
		}
	}

	for i := range datum.Fisher {
		sum := 0.0
		for _, v := range datum.Fisher[i].Measures {
			sum += v * v
		}
		sum = math.Sqrt(sum)
		for j := range datum.Fisher[i].Measures {
			datum.Fisher[i].Measures[j] /= sum
		}
	}

	if *FlagMeta {
		Meta(datum)
		return
	}

	if *FlagFull {
		Full(datum)
		return
	}

	pairs := make([][]iris.Iris, 0, 8)
	used := make(map[int]bool)
	for i, a := range datum.Fisher {
		if used[i] {
			continue
		}
		index, c, min := 0, iris.Iris{}, math.MaxFloat64
		for j := i + 1; j < len(datum.Fisher); j++ {
			if used[j] {
				continue
			}
			b := datum.Fisher[j]
			matrix := NewMatrix(0, 4, 2)
			matrix.Data = append(matrix.Data, a.Measures...)
			matrix.Data = append(matrix.Data, b.Measures...)
			entropy := SelfEntropy(matrix, matrix, matrix)
			diff := entropy[0] - entropy[1]
			if diff < 0 {
				diff = -diff
			}
			if diff < min {
				min = diff
				c = b
				index = j
			}
		}
		used[index] = true
		pairs = append(pairs, []iris.Iris{a, c})
	}
	correct := 0
	for i, pair := range pairs {
		fmt.Println(i, pair[0].Label, pair[1].Label, pair[0].Measures, pair[1].Measures)
		if pair[0].Label == pair[1].Label {
			correct++
		}
	}
	fmt.Println("correct", correct, "total", len(pairs), "percent", float64(correct)/float64(len(pairs)))
}
