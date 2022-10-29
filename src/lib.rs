use num;
use std::convert::TryInto;
use std::ops::{Add, Mul, Sub};

#[derive(Debug, PartialEq)]
pub enum TensorError {
    Rank,
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
    fn rank(&self) -> i32;
}

pub trait TensorOps {
    type Item;

    fn first(self) -> TensorResult<Self::Item>;
    fn less_than(self, other: TensorResult<Self::Item>) -> TensorResult<i32>;
    fn reshape(self, shape: Vec<i32>) -> TensorResult<Self::Item>;
    fn reverse(self, rank: Rank) -> TensorResult<Self::Item>;
}

pub trait TensorIntOps {
    // Unary Functions
    fn iota(self) -> TensorResult<i32>;

    // Unary Scalar Functions
    fn sign(self) -> TensorResult<i32>;

    // HOFs
    fn outer_product(
        self,
        other: TensorResult<i32>,
        binop: &dyn Fn(i32, i32) -> i32,
    ) -> TensorResult<i32>;
    fn reduce(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32>;
    fn scan(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32>;

    // Reduce Specializations
    fn maximum(self, rank: Rank) -> TensorResult<i32>;
    fn product(self, rank: Rank) -> TensorResult<i32>;
    fn sum(self, rank: Rank) -> TensorResult<i32>;
}

impl<T> TensorFns for Tensor<T> {
    fn rank(&self) -> i32 {
        self.shape.len() as i32
    }
}

impl<T: std::cmp::PartialOrd<T>> TensorOps for TensorResult<T> {
    type Item = T;

    fn first(self) -> TensorResult<T> {
        self.and_then(|t| {
            Ok(Tensor {
                shape: vec![],
                data: t.data.into_iter().take(1).collect(),
            })
        })
    }

    fn less_than(self, other: TensorResult<T>) -> TensorResult<i32> {
        self.and_then(|t| {
            other.and_then(|o| {
                let n = o.data.first().unwrap();
                Ok(Tensor {
                    shape: t.shape,
                    data: t
                        .data
                        .into_iter()
                        .map(|x| if x < *n { 1 } else { 0 })
                        .collect(),
                })
            })
        })
    }

    fn reshape(self, shape: Vec<i32>) -> TensorResult<T> {
        self.and_then(|t| {
            let n: i32 = shape.clone().iter().product();
            Ok(Tensor {
                shape,
                data: t.data.into_iter().take(n as usize).collect(),
            })
        })
    }

    fn reverse(self, rank: Rank) -> TensorResult<T> {
        self.and_then(|t| match rank {
            Some(_) => Err(TensorError::NotImplementedYet),
            None => Ok(Tensor {
                shape: t.shape,
                data: t.data.into_iter().rev().collect(),
            }),
        })
    }
}

impl TensorIntOps for TensorResult<i32> {
    fn iota(self) -> TensorResult<i32> {
        self.and_then(|t| {
            if t.rank() > 1 {
                return Err(TensorError::Rank);
            }
            let n: i32 = t.data.first().unwrap() + 1;
            Ok(build_vector((1..n).collect()))
        })
    }

    fn product(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&Mul::mul, rank)
    }

    fn outer_product(
        self,
        other: TensorResult<i32>,
        binop: &dyn Fn(i32, i32) -> i32,
    ) -> TensorResult<i32> {
        self.and_then(|t| {
            other.and_then(|o| {
                if t.rank() > 1 || o.rank() > 1 {
                    return Err(TensorError::Rank);
                }
                let rows = t.shape.first().unwrap();
                let cols = o.shape.first().unwrap();
                let new_data = t
                    .data
                    .into_iter()
                    .flat_map(|x| o.data.iter().map(move |y| binop(x, *y)))
                    .collect::<Vec<_>>();
                Ok(Tensor {
                    shape: vec![*rows, *cols],
                    data: new_data,
                })
            })
        })
    }

    fn reduce(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32> {
        self.and_then(|t| match rank {
            None => Ok(Tensor {
                shape: vec![],
                data: vec![t.data.into_iter().reduce(binop).unwrap()],
            }),
            Some(1) => {
                // TODO: only works for matrices
                let new_shape: Vec<i32> = t.shape.clone().into_iter().skip(1).collect();
                let chunk_size = t.shape.into_iter().nth(1).unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: t
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
                let new_shape: Vec<i32> = t.shape.clone().into_iter().take(1).collect();
                let chunk_size = t.shape.into_iter().nth(1).unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: t
                        .data
                        .chunks(chunk_size)
                        .map(|chunk| chunk.iter().copied().reduce(binop).unwrap())
                        .collect(),
                })
            }
            Some(_) => Err(TensorError::NotImplementedYet),
        })
    }

    fn scan(self, binop: &dyn Fn(i32, i32) -> i32, rank: Rank) -> TensorResult<i32> {
        self.and_then(|t| match rank {
            None => {
                let first = t.data.iter().copied().next().unwrap();
                Ok(Tensor {
                    shape: t.shape,
                    data: Some(first)
                        .into_iter()
                        .chain(t.data.into_iter().skip(1).scan(first, |acc, x| {
                            *acc = binop(*acc, x);
                            Some(*acc)
                        }))
                        .collect(),
                })
            }
            Some(_) => Err(TensorError::NotImplementedYet),
        })
    }

    fn sign(self) -> TensorResult<i32> {
        self.and_then(|t| {
            Ok(Tensor {
                shape: t.shape,
                data: t.data.into_iter().map(num::signum).collect(),
            })
        })
    }

    fn sum(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&Add::add, rank)
    }

    fn maximum(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&std::cmp::max, rank)
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
    let n = Ok(build_scalar(0));
    Ok(nums).less_than(n).sum(None)
}

// pub fn count_negatives(nums: Tensor) {
//     nums.lt(0).sum()
// }

pub fn max_wealth(accounts: Tensor<i32>) -> TensorResult<i32> {
    Ok(accounts).sum(Some(2)).maximum(None)
}

// pub fn max_wealth(accounts: Tensor) {
//     accounts.sum(1).maximum()
// }

pub fn array_sign(arr: Tensor<i32>) -> TensorResult<i32> {
    Ok(arr).sign().product(None)
}

// pub fn array_sign(arr: Tensor<i32>) {
//     arr.sign().product()
// }

pub fn mco(vec: Tensor<i32>) -> TensorResult<i32> {
    let op = |a, b| b * (a + b);
    Ok(vec).scan(&op, None).maximum(None)
}

// pub fn mco(Tensor vector) {
//     vector.scan(phi1(left, mul, plus))
//           .maximum()
// }

// pub fn check_matrix(Tensor grid) {
//     grid.eye()
//         .s(rev, id)
//         .equal(grid.min(1))
// }

// pub fn max_paren_depth(str equation) {
//     equation.to_tensor()
//             .outer("()")
//             .reduce(num::minus, 1)
//             .scan(num::plus)
//             .maximum()
// }

pub fn stringless_max_paren_depth(equation: Tensor<i32>) -> TensorResult<i32> {
    let rhs = Ok(build_vector(vec![2, 3]));
    Ok(equation)
        .outer_product(rhs, &|a, b| (a == b).into())
        .reduce(&Sub::sub, Some(2))
        .scan(&Add::add, None)
        .maximum(None)
}

pub fn smaller_numbers_than_current(nums: Tensor<i32>) -> TensorResult<i32> {
    Ok(nums.clone())
        .outer_product(Ok(nums), &|a, b| (a > b).into())
        .sum(Some(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first() {
        {
            let input = Ok(build_vector(vec![1, 2, 3]));
            let expected = Ok(build_scalar(1));
            assert_eq!(input.first(), expected);
        }
    }

    #[test]
    fn test_matrix_sums() {
        {
            // matrix sum
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_scalar(45));
            assert_eq!(input.sum(None), expected);
        }
        {
            // column sums
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_vector(vec![12, 15, 18]));
            assert_eq!(input.sum(Some(1)), expected);
        }
        {
            // row sums
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_vector(vec![6, 15, 24]));
            assert_eq!(input.sum(Some(2)), expected);
        }
    }

    #[test]
    fn test_matrix_maximums() {
        {
            // matrix maximum
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_scalar(9));
            assert_eq!(input.maximum(None), expected);
        }
        {
            // column maximums
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_vector(vec![7, 8, 9]));
            assert_eq!(input.maximum(Some(1)), expected);
        }
        {
            // row maximums
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_vector(vec![3, 6, 9]));
            assert_eq!(input.maximum(Some(2)), expected);
        }
    }

    #[test]
    fn test_matrix_iota() {
        let iota_matrix = Ok(build_scalar(12)).iota().reshape(vec![3, 4]);
        let expected = Ok(build_matrix(
            vec![3, 4],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ));
        assert_eq!(iota_matrix, expected);
    }

    #[test]
    fn test_count_negatives() {
        // https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/
        {
            let input = build_matrix(vec![2, 2], vec![-1, -2, 3, 4]);
            let expected = Ok(build_scalar(2));
            assert_eq!(count_negatives(input), expected);
        }
        {
            let input = build_matrix(
                vec![4, 4],
                vec![4, 3, 2, -1, 3, 2, 1, -1, 1, 1, -1, -2, -1, -1, -2, -3],
            );
            let expected = Ok(build_scalar(8));
            assert_eq!(count_negatives(input), expected);
        }
    }

    #[test]
    fn test_max_wealth() {
        // https://leetcode.com/problems/richest-customer-wealth/
        {
            let input = build_matrix(vec![2, 3], vec![1, 2, 3, 3, 2, 1]);
            let expected = Ok(build_scalar(6));
            assert_eq!(max_wealth(input), expected);
        }
        {
            let input = build_matrix(vec![3, 2], vec![1, 5, 7, 3, 3, 5]);
            let expected = Ok(build_scalar(10));
            assert_eq!(max_wealth(input), expected);
        }
        {
            let input = build_matrix(vec![3, 3], vec![2, 8, 7, 7, 1, 3, 1, 9, 5]);
            let expected = Ok(build_scalar(17));
            assert_eq!(max_wealth(input), expected);
        }
    }

    #[test]
    fn test_array_sign() {
        // https://leetcode.com/problems/sign-of-the-product-of-an-array/
        {
            let input = build_vector(vec![-1, -2, -3, -4, 3, 2, 1]);
            let expected = Ok(build_scalar(1));
            assert_eq!(array_sign(input), expected);
        }
        {
            let input = build_vector(vec![1, 5, 0, 2, -3]);
            let expected = Ok(build_scalar(0));
            assert_eq!(array_sign(input), expected);
        }
        {
            let input = build_vector(vec![-1, 1, -1, 1, -1]);
            let expected = Ok(build_scalar(-1));
            assert_eq!(array_sign(input), expected);
        }
    }

    #[test]
    fn test_mco() {
        // https://leetcode.com/problems/max-consecutive-ones/
        {
            let input = build_vector(vec![1, 1, 0, 1, 1, 1]);
            let expected = Ok(build_scalar(3));
            assert_eq!(mco(input), expected);
        }
        {
            let input = build_vector(vec![1, 0, 1, 1, 0, 1]);
            let expected = Ok(build_scalar(2));
            assert_eq!(mco(input), expected);
        }
    }

    #[test]
    fn test_outer_product() {
        let lhs = build_vector(vec![1, 2, 3]);
        let rhs = build_vector(vec![1, 2, 3]);
        let expected = Ok(build_matrix(vec![3, 3], vec![2, 3, 4, 3, 4, 5, 4, 5, 6]));
        assert_eq!(Ok(lhs).outer_product(Ok(rhs), &Add::add), expected);
    }

    #[test]
    fn test_stringless_max_paren_depth() {
        let input = build_vector(vec![2, 2, 4, 5, 3, 2, 2, 3, 3, 8, 3, 3]);
        let expected = Ok(build_scalar(3));
        assert_eq!(stringless_max_paren_depth(input), expected);
    }

    #[test]
    fn test_smaller_numbers_than_current() {
        // https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/
        {
            let input = build_vector(vec![8, 1, 2, 2, 3]);
            let expected = Ok(build_vector(vec![4, 0, 1, 1, 3]));
            assert_eq!(smaller_numbers_than_current(input), expected);
        }
        {
            let input = build_vector(vec![6, 5, 4, 8]);
            let expected = Ok(build_vector(vec![2, 1, 0, 3]));
            assert_eq!(smaller_numbers_than_current(input), expected);
        }
        {
            let input = build_vector(vec![7, 7, 7, 7]);
            let expected = Ok(build_vector(vec![0, 0, 0, 0]));
            assert_eq!(smaller_numbers_than_current(input), expected);
        }
    }

    // TODO:
    // https://leetcode.com/problems/k-diff-pairs-in-an-array/
}
