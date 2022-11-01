use itertools::Itertools;
use std::collections::HashSet;
use std::convert::TryInto;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, PartialEq)]
pub enum TensorError {
    Rank,
    Shape,
    Type,
    NotImplementedYet,
}

type Rank = Option<i32>;

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T> {
    shape: Vec<i32>,
    data: Vec<T>,
}

type TensorResult<T> = Result<Tensor<T>, TensorError>;

pub trait TensorFns {
    fn len(&self) -> Tensor<i32>;
    fn rank(&self) -> i32;
}

pub trait TensorOps {
    type Item;

    fn first(self) -> Tensor<Self::Item>;
    fn intersection(self, other: Tensor<Self::Item>) -> TensorResult<Self::Item>;
    fn reshape(self, shape: Vec<i32>) -> Tensor<Self::Item>;
    fn reverse(self, rank: Rank) -> TensorResult<Self::Item>;
    fn sort(self) -> TensorResult<Self::Item>;
    fn unique(self) -> TensorResult<Self::Item>;

    // Binary Scalar Functions
    fn scalar_binary_operation(
        self,
        other: Tensor<Self::Item>,
        binop: &dyn Fn(Self::Item, Self::Item) -> i32,
    ) -> TensorResult<i32>;
    fn equal(self, other: Tensor<Self::Item>) -> TensorResult<i32>;
    fn less_than(self, other: Tensor<Self::Item>) -> TensorResult<i32>;
    fn less_than_or_equal(self, other: Tensor<Self::Item>) -> TensorResult<i32>;
}

pub trait TensorIntOps {
    // Unary Functions
    fn indices(self) -> TensorResult<i32>;
    fn iota(self) -> Tensor<i32>;

    // Unary Scalar Functions
    fn sign(self) -> TensorResult<i32>;

    // Binary Scalar Functions
    fn min(self, other: Tensor<i32>) -> TensorResult<i32>;
    fn multiply(self, other: Tensor<i32>) -> TensorResult<i32>;

    // HOFs
    fn outer_product(
        self,
        other: Tensor<i32>,
        binop: &dyn Fn(i32, i32) -> i32,
    ) -> TensorResult<i32>;
    fn reduce(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32>;
    fn scan(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32>;
    fn triangle_product(self, binop: &dyn Fn(i32, i32) -> i32) -> TensorResult<i32>;
    fn windowed_reduce(
        self,
        window_size: usize,
        binop: &dyn Fn(i32, i32) -> i32,
    ) -> TensorResult<i32>;

    // Reduce Specializations
    fn any(self, rank: Rank) -> TensorResult<i32>;
    fn maximum(self, rank: Rank) -> TensorResult<i32>;
    fn minimum(self, rank: Rank) -> TensorResult<i32>;
    fn product(self, rank: Rank) -> TensorResult<i32>;
    fn sum(self, rank: Rank) -> TensorResult<i32>;
}

impl<T> TensorFns for Tensor<T> {
    fn len(&self) -> Tensor<i32> {
        Tensor {
            shape: vec![],
            data: vec![*self.shape.first().unwrap()],
        }
    }

    fn rank(&self) -> i32 {
        self.shape.len() as i32
    }
}

impl<
        T: std::cmp::PartialOrd<T>
            + std::hash::Hash
            + std::cmp::Eq
            + std::clone::Clone
            + std::marker::Copy
            + std::cmp::Ord,
    > TensorOps for Tensor<T>
{
    type Item = T;

    fn first(self) -> Tensor<T> {
        Tensor {
            shape: vec![],
            data: self.data.into_iter().take(1).collect(),
        }
    }

    fn intersection(self, other: Tensor<Self::Item>) -> TensorResult<Self::Item> {
        if self.rank() > 1 {
            return Err(TensorError::Rank);
        }
        let new_data = self
            .data
            .into_iter()
            .collect::<HashSet<_>>()
            .intersection(&other.data.into_iter().collect::<HashSet<_>>())
            .copied()
            .collect::<Vec<_>>();
        Ok(Tensor {
            shape: vec![new_data.len() as i32],
            data: new_data,
        })
    }

    fn scalar_binary_operation(
        self,
        other: Tensor<T>,
        binop: &dyn Fn(T, T) -> i32,
    ) -> TensorResult<i32> {
        if other.rank() == 0 {
            let n = other.data.first().unwrap();
            return Ok(Tensor {
                shape: self.shape,
                data: self.data.into_iter().map(|x| binop(x, *n)).collect(),
            });
        }
        if self.rank() == 0 {
            return Err(TensorError::NotImplementedYet);
        }
        if self.shape == other.shape {
            return Ok(Tensor {
                shape: self.shape,
                data: self
                    .data
                    .into_iter()
                    .zip(other.data.into_iter())
                    .map(|(a, b)| binop(a, b))
                    .collect(),
            });
        }
        Err(TensorError::Shape)
    }

    fn equal(self, other: Tensor<T>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &|a, b| (a == b).into())
    }

    fn less_than(self, other: Tensor<T>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &|a, b| (a < b).into())
    }

    fn less_than_or_equal(self, other: Tensor<T>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &|a, b| (a <= b).into())
    }

    fn reshape(self, shape: Vec<i32>) -> Tensor<T> {
        let n: i32 = shape.clone().iter().product();
        Tensor {
            shape,
            data: self.data.into_iter().take(n as usize).collect(),
        }
    }

    fn reverse(self, rank: Rank) -> TensorResult<T> {
        match rank {
            Some(_) => Err(TensorError::NotImplementedYet),
            None => Ok(Tensor {
                shape: self.shape,
                data: self.data.into_iter().rev().collect(),
            }),
        }
    }

    fn sort(self) -> TensorResult<T> {
        if self.rank() > 1 {
            return Err(TensorError::NotImplementedYet);
        }
        Ok(Tensor {
            shape: self.shape,
            data: self.data.into_iter().sorted().collect(),
        })
    }

    fn unique(self) -> TensorResult<T> {
        if self.rank() > 1 {
            return Err(TensorError::Rank);
        }
        let new_data: Vec<_> = self.data.into_iter().unique().collect();
        Ok(Tensor {
            shape: vec![new_data.len() as i32],
            data: new_data,
        })
    }
}

impl TensorIntOps for Tensor<i32> {
    fn indices(self) -> TensorResult<i32> {
        if self.rank() != 1 {
            return Err(TensorError::NotImplementedYet);
        }
        let new_data = self
            .data
            .into_iter()
            .enumerate()
            .filter(|(_, x)| *x == 1)
            .map(|(i, _)| i as i32 + 1)
            .collect::<Vec<_>>();
        Ok(Tensor {
            shape: vec![new_data.len() as i32],
            data: new_data,
        })
    }

    fn iota(self) -> Tensor<i32> {
        if self.rank() == 0 {
            let n: i32 = self.data.first().unwrap() + 1;
            return build_vector((1..n).collect());
        } else if self.rank() == 1 {
            let n = self.data.clone().into_iter().product();
            return Tensor {
                shape: self.data,
                data: (1..=n).collect(),
            };
        }
        // TODO: implement here
        // don't return TensorError::NotImplementedYet so i can in tests (without ?)
        build_scalar(-1)
    }

    fn min(self, other: Tensor<i32>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &std::cmp::min)
    }

    fn multiply(self, other: Tensor<i32>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &Mul::mul)
    }

    fn outer_product(
        self,
        other: Tensor<i32>,
        binop: &dyn Fn(i32, i32) -> i32,
    ) -> TensorResult<i32> {
        if self.rank() > 1 || other.rank() > 1 {
            return Err(TensorError::Rank);
        }
        let rows = self.shape.first().unwrap();
        let cols = other.shape.first().unwrap();
        Ok(Tensor {
            shape: vec![*rows, *cols],
            data: self
                .data
                .into_iter()
                .flat_map(|x| other.data.iter().map(move |y| binop(x, *y)))
                .collect::<Vec<_>>(),
        })
    }

    fn triangle_product(self, binop: &dyn Fn(i32, i32) -> i32) -> TensorResult<i32> {
        if self.rank() > 1 {
            return Err(TensorError::Rank);
        }
        let n = self.shape.first().unwrap();
        Ok(Tensor {
            shape: vec![*n, *n],
            data: self
                .data
                .iter()
                .copied()
                .enumerate()
                .flat_map(|(i, x)| {
                    self.data
                        .iter()
                        .enumerate()
                        .map(move |(j, y)| if j > i { binop(x, *y) } else { 0 })
                })
                .collect::<Vec<_>>(),
        })
    }

    fn windowed_reduce(
        self,
        window_size: usize,
        binop: &dyn Fn(i32, i32) -> i32,
    ) -> TensorResult<i32> {
        if self.rank() != 1 {
            return Err(TensorError::Rank);
        }
        Ok(Tensor {
            shape: vec![(self.data.len() - window_size + 1) as i32],
            data: self
                .data
                .windows(window_size)
                .map(|w| w.into_iter().copied().reduce(binop).unwrap())
                .collect(),
        })
    }

    fn reduce(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32> {
        match rank {
            None => Ok(Tensor {
                shape: vec![],
                data: vec![self.data.into_iter().reduce(binop).unwrap()],
            }),
            Some(1) => {
                // TODO: only works for matrices
                let new_shape: Vec<i32> = self.shape.clone().into_iter().skip(1).collect();
                let chunk_size = self.shape.into_iter().nth(1).unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .fold(vec![0; chunk_size], |acc, chunk| {
                            acc.iter()
                                .zip(chunk.iter())
                                .map(|(a, b)| binop(*a, *b))
                                .collect()
                        }),
                })
            }
            Some(2) => {
                // TODO: only works for matrices
                let new_shape: Vec<i32> = self.shape.clone().into_iter().take(1).collect();
                let chunk_size = self.shape.into_iter().nth(1).unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .map(|chunk| chunk.iter().copied().reduce(binop).unwrap())
                        .collect(),
                })
            }
            Some(_) => Err(TensorError::NotImplementedYet),
        }
    }

    fn scan(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32> {
        match rank {
            None => {
                let first = self.data.iter().copied().next().unwrap();
                Ok(Tensor {
                    shape: self.shape,
                    data: Some(first)
                        .into_iter()
                        .chain(self.data.into_iter().skip(1).scan(first, |acc, x| {
                            *acc = binop(*acc, x);
                            Some(*acc)
                        }))
                        .collect(),
                })
            }
            Some(_) => Err(TensorError::NotImplementedYet),
        }
    }

    fn sign(self) -> TensorResult<i32> {
        Ok(Tensor {
            shape: self.shape,
            data: self.data.into_iter().map(num::signum).collect(),
        })
    }

    fn any(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&std::cmp::max, rank)?.min(build_scalar(1))
    }

    fn product(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&Mul::mul, rank)
    }

    fn maximum(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&std::cmp::max, rank)
    }

    fn minimum(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&std::cmp::min, rank)
    }

    fn sum(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&Add::add, rank)
    }
}

pub fn build_scalar(data: i32) -> Tensor<i32> {
    Tensor {
        shape: vec![],
        data: vec![data],
    }
}

pub fn build_vector<T>(data: Vec<T>) -> Tensor<T> {
    Tensor {
        shape: vec![data.len() as i32],
        data,
    }
}

pub fn build_matrix<T>(shape: Vec<i32>, data: Vec<T>) -> Tensor<T> {
    Tensor { shape, data }
}

pub fn print_tensor(tr: TensorResult<i32>) {
    match tr {
        Err(e) => println!("{:?}", e),
        Ok(t) => {
            if t.rank() == 0 {
                println!("Scalar\n{:?}", t.data);
            } else if t.rank() == 1 {
                println!("Vector\n{:?}", t.data);
            } else if t.rank() == 2 {
                println!("Matrix");
                let n: usize = (*t.shape.first().unwrap()).try_into().unwrap();
                for chunk in t.data.chunks(n) {
                    println!("{:?}", chunk);
                }
            } else {
                println!("Not implemented yet");
            }
        }
    }
}

pub fn count_negatives(nums: Tensor<i32>) -> TensorResult<i32> {
    let n = build_scalar(0);
    nums.less_than(n)?.sum(None)
}

// pub fn count_negatives(nums: Tensor) {
//     nums.lt(0).sum()
// }

pub fn max_wealth(accounts: Tensor<i32>) -> TensorResult<i32> {
    accounts.sum(Some(2))?.maximum(None)
}

// // pub fn max_wealth(accounts: Tensor) {
// //     accounts.sum(1).maximum()
// // }

pub fn array_sign(arr: Tensor<i32>) -> TensorResult<i32> {
    arr.sign()?.product(None)
}

// // pub fn array_sign(arr: Tensor<i32>) {
// //     arr.sign().product()
// // }

pub fn mco(vec: Tensor<i32>) -> TensorResult<i32> {
    let op = |a, b| b * (a + b);
    vec.scan(&op, None)?.maximum(None)
}

// // pub fn mco(Tensor vector) {
// //     vector.scan(phi1(left, mul, plus))
// //           .maximum()
// // }

// // pub fn check_matrix(Tensor grid) {
// //     grid.eye()
// //         .s(rev, id)
// //         .equal(grid.min(1))
// // }

// // pub fn max_paren_depth(str equation) {
// //     equation.to_tensor()
// //             .outer("()")
// //             .reduce(num::minus, 1)
// //             .scan(num::plus)
// //             .maximum()
// // }

pub fn stringless_max_paren_depth(equation: Tensor<i32>) -> TensorResult<i32> {
    let rhs = build_vector(vec![2, 3]);
    equation
        .outer_product(rhs, &|a, b| (a == b).into())?
        .reduce(&Sub::sub, Some(2))?
        .scan(&Add::add, None)?
        .maximum(None)
}

pub fn smaller_numbers_than_current(nums: Tensor<i32>) -> TensorResult<i32> {
    nums.clone()
        .outer_product(nums, &|a, b| (a > b).into())?
        .sum(Some(2))
}

pub fn find_pairs(nums: Tensor<i32>, k: i32) -> TensorResult<i32> {
    let uniq = nums.unique()?;
    uniq.clone()
        .outer_product(uniq, &Sub::sub)?
        .equal(build_scalar(k))?
        .sum(None)
}

pub fn lucky_numbers(matrix: Tensor<i32>) -> TensorResult<i32> {
    matrix
        .clone()
        .minimum(Some(2))?
        .intersection(matrix.maximum(Some(1))?)
}

pub fn num_special(mat: Tensor<i32>) -> TensorResult<i32> {
    mat.clone()
        .sum(Some(1))?
        .multiply(mat.sum(Some(2))?)?
        .equal(build_scalar(1))?
        .sum(None)
}

pub fn num_identical_pairs(nums: Tensor<i32>) -> TensorResult<i32> {
    nums.triangle_product(&|a, b| (a == b).into())?.sum(None)
}

pub fn max_ice_cream(costs: Tensor<i32>, coins: Tensor<i32>) -> TensorResult<i32> {
    costs
        .sort()?
        .scan(&Add::add, None)?
        .less_than_or_equal(coins)?
        .sum(None)
}

pub fn can_make_arithmetic_progression(arr: Tensor<i32>) -> TensorResult<i32> {
    arr.sort()?
        .windowed_reduce(2, &Sub::sub)?
        .unique()?
        .len()
        .equal(build_scalar(1))
}

pub fn first_uniq_num(nums: Tensor<i32>) -> TensorResult<i32> {
    Ok(nums
        .clone()
        .outer_product(nums, &|a, b| (a == b).into())?
        .sum(Some(2))?
        .equal(build_scalar(1))?
        .indices()?
        .first())
}

pub fn check_if_double_exists(arr: Tensor<i32>) -> TensorResult<i32> {
    arr.triangle_product(&|a, b| (a == 2 * b).into())?.any(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first() {
        {
            let input = build_vector(vec![1, 2, 3]);
            let expected = build_scalar(1);
            assert_eq!(input.first(), expected);
        }
    }

    #[test]
    fn test_iota() {
        {
            let input = build_scalar(3);
            let expected = build_vector(vec![1, 2, 3]);
            assert_eq!(input.iota(), expected);
        }
    }

    #[test]
    fn test_reverse() {
        {
            let input = build_scalar(3).iota();
            let expected = build_vector(vec![3, 2, 1]);
            assert_eq!(input.reverse(None).unwrap(), expected);
        }
    }

    #[test]
    fn test_matrix_sums() {
        {
            // matrix sum
            let input = build_vector(vec![3, 3]).iota();
            let expected = build_scalar(45);
            assert_eq!(input.sum(None).unwrap(), expected);
        }
        {
            // column sums
            let input = build_vector(vec![3, 3]).iota();
            let expected = build_vector(vec![12, 15, 18]);
            assert_eq!(input.sum(Some(1)).unwrap(), expected);
        }
        {
            // row sums
            let input = build_vector(vec![3, 3]).iota();
            let expected = build_vector(vec![6, 15, 24]);
            assert_eq!(input.sum(Some(2)).unwrap(), expected);
        }
    }

    #[test]
    fn test_matrix_maximums() {
        {
            // matrix maximum
            let input = build_vector(vec![3, 3]).iota();
            let expected = build_scalar(9);
            assert_eq!(input.maximum(None).unwrap(), expected);
        }
        {
            // column maximums
            let input = build_vector(vec![3, 3]).iota();
            let expected = build_vector(vec![7, 8, 9]);
            assert_eq!(input.maximum(Some(1)).unwrap(), expected);
        }
        {
            // row maximums
            let input = build_vector(vec![3, 3]).iota();
            let expected = build_vector(vec![3, 6, 9]);
            assert_eq!(input.maximum(Some(2)).unwrap(), expected);
        }
    }

    #[test]
    fn test_count_negatives() {
        // https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/
        {
            let input = build_matrix(vec![2, 2], vec![-1, -2, 3, 4]);
            let expected = build_scalar(2);
            assert_eq!(count_negatives(input).unwrap(), expected);
        }
        {
            let input = build_matrix(
                vec![4, 4],
                vec![4, 3, 2, -1, 3, 2, 1, -1, 1, 1, -1, -2, -1, -1, -2, -3],
            );
            let expected = build_scalar(8);
            assert_eq!(count_negatives(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_max_wealth() {
        // https://leetcode.com/problems/richest-customer-wealth/
        {
            let input = build_matrix(vec![2, 3], vec![1, 2, 3, 3, 2, 1]);
            let expected = build_scalar(6);
            assert_eq!(max_wealth(input).unwrap(), expected);
        }
        {
            let input = build_matrix(vec![3, 2], vec![1, 5, 7, 3, 3, 5]);
            let expected = build_scalar(10);
            assert_eq!(max_wealth(input).unwrap(), expected);
        }
        {
            let input = build_matrix(vec![3, 3], vec![2, 8, 7, 7, 1, 3, 1, 9, 5]);
            let expected = build_scalar(17);
            assert_eq!(max_wealth(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_array_sign() {
        // https://leetcode.com/problems/sign-of-the-product-of-an-array/
        {
            let input = build_vector(vec![-1, -2, -3, -4, 3, 2, 1]);
            let expected = build_scalar(1);
            assert_eq!(array_sign(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![1, 5, 0, 2, -3]);
            let expected = build_scalar(0);
            assert_eq!(array_sign(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![-1, 1, -1, 1, -1]);
            let expected = build_scalar(-1);
            assert_eq!(array_sign(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_mco() {
        // https://leetcode.com/problems/max-consecutive-ones/
        {
            let input = build_vector(vec![1, 1, 0, 1, 1, 1]);
            let expected = build_scalar(3);
            assert_eq!(mco(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![1, 0, 1, 1, 0, 1]);
            let expected = build_scalar(2);
            assert_eq!(mco(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_outer_product() {
        let lhs = build_vector(vec![1, 2, 3]);
        let rhs = build_vector(vec![1, 2, 3]);
        let expected = build_matrix(vec![3, 3], vec![2, 3, 4, 3, 4, 5, 4, 5, 6]);
        assert_eq!(lhs.outer_product(rhs, &Add::add).unwrap(), expected);
    }

    #[test]
    fn test_stringless_max_paren_depth() {
        let input = build_vector(vec![2, 2, 4, 5, 3, 2, 2, 3, 3, 8, 3, 3]);
        let expected = build_scalar(3);
        assert_eq!(stringless_max_paren_depth(input).unwrap(), expected);
    }

    #[test]
    fn test_smaller_numbers_than_current() {
        // https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/
        {
            let input = build_vector(vec![8, 1, 2, 2, 3]);
            let expected = build_vector(vec![4, 0, 1, 1, 3]);
            assert_eq!(smaller_numbers_than_current(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![6, 5, 4, 8]);
            let expected = build_vector(vec![2, 1, 0, 3]);
            assert_eq!(smaller_numbers_than_current(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![7, 7, 7, 7]);
            let expected = build_vector(vec![0, 0, 0, 0]);
            assert_eq!(smaller_numbers_than_current(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_find_pairs() {
        // https://leetcode.com/problems/k-diff-pairs-in-an-array/
        {
            let input = build_vector(vec![3, 1, 4, 1, 5]);
            let expected = build_scalar(2);
            assert_eq!(find_pairs(input, 2).unwrap(), expected);
        }
        {
            let input = build_vector((1..=5).collect());
            let expected = build_scalar(4);
            assert_eq!(find_pairs(input, 1).unwrap(), expected);
        }
    }

    #[test]
    fn test_lucky_numbers() {
        // https://leetcode.com/problems/lucky-numbers-in-a-matrix/
        {
            let input = build_matrix(vec![3, 3], vec![3, 7, 8, 9, 11, 13, 15, 16, 17]);
            let expected = build_vector(vec![15]);
            assert_eq!(lucky_numbers(input).unwrap(), expected);
        }
        {
            let input = build_matrix(vec![4, 4], vec![1, 10, 4, 2, 9, 3, 8, 7, 15, 16, 17, 12]);
            let expected = build_vector(vec![12]);
            assert_eq!(lucky_numbers(input).unwrap(), expected);
        }
        {
            let input = build_matrix(vec![2, 2], vec![7, 8, 1, 2]);
            let expected = build_vector(vec![7]);
            assert_eq!(lucky_numbers(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_num_special() {
        // https://leetcode.com/problems/special-positions-in-a-binary-matrix/
        {
            let input = build_matrix(vec![3, 3], vec![1, 0, 0, 0, 0, 1, 1, 0, 0]);
            let expected = build_scalar(1);
            assert_eq!(num_special(input).unwrap(), expected);
        }
        {
            let input = build_matrix(vec![3, 3], vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
            let expected = build_scalar(3);
            assert_eq!(num_special(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_num_identical_pairs() {
        // https://leetcode.com/problems/number-of-good-pairs/
        {
            let input = build_vector(vec![1, 2, 3, 1, 1, 3]);
            let expected = build_scalar(4);
            assert_eq!(num_identical_pairs(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![1, 1, 1, 1]);
            let expected = build_scalar(6);
            assert_eq!(num_identical_pairs(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![1, 2, 3]);
            let expected = build_scalar(0);
            assert_eq!(num_identical_pairs(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_max_ice_cream() {
        // https://leetcode.com/problems/maximum-ice-cream-bars/
        {
            let input = build_vector(vec![1, 3, 2, 4, 1]);
            let expected = build_scalar(4);
            assert_eq!(max_ice_cream(input, build_scalar(7)).unwrap(), expected);
        }
        {
            let input = build_vector(vec![10, 6, 8, 7, 7, 8]);
            let expected = build_scalar(0);
            assert_eq!(max_ice_cream(input, build_scalar(5)).unwrap(), expected);
        }
        {
            let input = build_vector(vec![1, 6, 3, 1, 2, 5]);
            let expected = build_scalar(6);
            assert_eq!(max_ice_cream(input, build_scalar(20)).unwrap(), expected);
        }
    }

    #[test]
    fn test_can_make_arithmetic_progression() {
        // https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence/
        {
            let input = build_vector(vec![3, 5, 1]);
            let expected = build_scalar(1);
            assert_eq!(can_make_arithmetic_progression(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![1, 2, 4]);
            let expected = build_scalar(0);
            assert_eq!(can_make_arithmetic_progression(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_first_unique_number_in_a_vector() {
        // modified from https://leetcode.com/problems/first-unique-character-in-a-string/
        {
            let input = build_vector(vec![1, 2, 2, 3, 4, 5, 6, 2]);
            let expected = build_scalar(1);
            assert_eq!(first_uniq_num(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_check_if_double_exists() {
        // https://leetcode.com/problems/check-if-n-and-its-double-exist/
        {
            let input = build_vector(vec![10, 2, 5, 3]);
            let expected = build_scalar(1);
            assert_eq!(check_if_double_exists(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![3, 1, 7, 11]);
            let expected = build_scalar(0);
            assert_eq!(check_if_double_exists(input).unwrap(), expected);
        }
    }
}
