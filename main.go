// Copyright 2023 The Ent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"sort"

	"github.com/bugra/kmeans"
	"github.com/pointlander/datum/iris"
)

const (
	// Width is the width of the vector
	Width = 4
)

// Meta is the meta data analysis mode
func Meta(fisher Matrix) {
	adjacency := NewMatrix(0, fisher.Rows, fisher.Rows)
	for i := 0; i < fisher.Rows; i++ {
		a := fisher.Data[i*Width : i*Width+Width]
		for j := 0; j < fisher.Rows; j++ {
			b := fisher.Data[j*Width : j*Width+Width]
			matrix := NewMatrix(0, Width, 2)
			matrix.Data = append(matrix.Data, a...)
			matrix.Data = append(matrix.Data, b...)
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
func Full(datum iris.Datum, fisher Matrix) {
	for i := 0; i < 2; i++ {
		original := fisher
		fisher = Normalize(PCA(fisher))
		fisher = SelfAttention(original, fisher, original)
	}
	entropy := Normalize(PCA(fisher)) //SelfAttention(fisher, fisher, fisher)
	type Entropy struct {
		Index int
		Value float64
	}
	entropyList := make([]Entropy, 0, len(entropy.Data))
	for i := 0; i < entropy.Rows; i++ {
		v := entropy.Data[i*entropy.Cols : i*entropy.Cols+entropy.Cols]
		entropyList = append(entropyList, Entropy{i, v[0]})
	}
	sort.Slice(entropyList, func(i, j int) bool {
		return entropyList[i].Value < entropyList[j].Value
	})
	for _, v := range entropyList {
		fmt.Printf("%s %.64f\n", datum.Fisher[v.Index].Label, v.Value)
	}
}

// SelfAware is the self aware mode
func SelfAware(datum iris.Datum, fisher Matrix) {
	a := Normalize(fisher)
	b := PCA(Append(a, T(PageRank(Abs(Mul(a, a))))))
	b = SelfAttention(b, b, b)
	type Vector struct {
		Index int
		Value []float64
	}
	vectors := make([]Vector, 0, len(b.Data))
	for i := 0; i < b.Rows; i++ {
		v := b.Data[i*b.Cols : i*b.Cols+b.Cols]
		vectors = append(vectors, Vector{i, v})
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
	clusters, err := kmeans.Kmeans(rawData, 3, kmeans.EuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Println(i, v)
	}
}

var (
	// FlagMeta meta mode
	FlagMeta = flag.Bool("meta", false, "meta mode")
	// FlagFull full mode
	FlagFull = flag.Bool("full", false, "full mode")
	// FlagSelfAware self aware mode
	FlagSelfAware = flag.Bool("self", false, "self aware mode")
)

func main() {
	flag.Parse()

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	fisher := NewMatrix(0, 4, len(datum.Fisher))
	for _, embedding := range datum.Fisher {
		fisher.Data = append(fisher.Data, embedding.Measures...)
	}

	if *FlagFull {
		Full(datum, fisher)
		return
	}

	if *FlagSelfAware {
		SelfAware(datum, fisher)
		return
	}

	fisher = Normalize(PCA(fisher))

	if *FlagMeta {
		Meta(fisher)
		return
	}

	type Pair struct {
		A, B    []float64
		SourceA iris.Iris
		SourceB iris.Iris
	}

	pairs := make([]Pair, 0, 8)
	used := make(map[int]bool)
	for i := 0; i < fisher.Rows; i++ {
		if used[i] {
			continue
		}
		a := fisher.Data[i*Width : i*Width+Width]
		index, c, min := 0, []float64{}, math.MaxFloat64
		for j := i + 1; j < fisher.Rows; j++ {
			if used[j] {
				continue
			}
			b := fisher.Data[j*Width : j*Width+Width]
			matrix := NewMatrix(0, Width, 2)
			matrix.Data = append(matrix.Data, a...)
			matrix.Data = append(matrix.Data, b...)
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
		pairs = append(pairs, Pair{
			A:       a,
			B:       c,
			SourceA: datum.Fisher[i],
			SourceB: datum.Fisher[index],
		})
	}
	correct := 0
	for i, pair := range pairs {
		fmt.Println(i, pair.SourceA.Label, pair.SourceB.Label, pair.SourceA.Measures, pair.SourceB.Measures)
		if pair.SourceA.Label == pair.SourceB.Label {
			correct++
		}
	}
	fmt.Println("correct", correct, "total", len(pairs), "percent", float64(correct)/float64(len(pairs)))
}
