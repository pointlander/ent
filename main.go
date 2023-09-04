// Copyright 2023 The Ent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"

	"github.com/pointlander/datum/iris"
)

func main() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
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
			sum := 0.0
			for _, v := range entropy {
				sum += v
			}
			sum = -sum
			//fmt.Println(i, j, sum)
			if sum < min {
				min = sum
				c = b
				index = j
			}
		}
		used[index] = true
		pairs = append(pairs, []iris.Iris{a, c})
	}
	for i, pair := range pairs {
		fmt.Println(i, pair[0].Label, pair[1].Label, pair[0].Measures, pair[1].Measures)
	}
}
