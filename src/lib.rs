use std::convert::TryInto;

#[derive(Debug, PartialEq)]
pub enum TensorError {
    Rank,
    Type,
    NotImplementedYet,
}

type Rank = Option<i32>;

#[derive(Debug, PartialEq)]
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
    fn iota(self) -> TensorResult<i32>;
    fn sum(self, rank: Rank) -> TensorResult<i32>;
    fn maximum(self, rank: Rank) -> TensorResult<i32>;
}

impl<T> TensorFns for Tensor<T> {
    fn rank(&self) -> i32 {
        self.shape.len() as i32
    }
}

impl<T: std::cmp::PartialOrd<T>> TensorOps for TensorResult<T> {
    type Item = T;

    fn first(self) -> TensorResult<T> {
        match self {
            Err(e) => Err(e),
            Ok(t) => Ok(Tensor {
                shape: vec![],
                data: t.data.into_iter().take(1).collect(),
            }),
        }
    }

    fn less_than(self, other: TensorResult<T>) -> TensorResult<i32> {
        match self {
            Err(e) => Err(e),
            Ok(t) => match other {
                Err(e) => Err(e),
                Ok(o) => {
                    let n = o.data.into_iter().next().unwrap();
                    Ok(Tensor {
                        shape: t.shape,
                        data: t
                            .data
                            .into_iter()
                            .map(|x| if x < n { 1 } else { 0 })
                            .collect(),
                    })
                }
            },
        }
    }

    fn reshape(self, shape: Vec<i32>) -> TensorResult<T> {
        match self {
            Err(e) => Err(e),
            Ok(t) => {
                let n: i32 = shape.clone().iter().product();
                Ok(Tensor {
                    shape,
                    data: t.data.into_iter().take(n as usize).collect(),
                })
            }
        }
    }

    fn reverse(self, rank: Rank) -> TensorResult<T> {
        match self {
            Err(e) => Err(e),
            Ok(t) => match rank {
                Some(_) => Err(TensorError::NotImplementedYet),
                None => Ok(Tensor {
                    shape: t.shape,
                    data: t.data.into_iter().rev().collect(),
                }),
            },
        }
    }
}

impl TensorIntOps for TensorResult<i32> {
    fn iota(self) -> TensorResult<i32> {
        match self {
            Err(e) => Err(e),
            Ok(t) => {
                if t.rank() > 1 {
                    return Err(TensorError::Rank);
                }
                let n: i32 = t.data.into_iter().next().unwrap() + 1;
                Ok(build_vector((1..n).collect()))
            }
        }
    }

    fn sum(self, rank: Rank) -> TensorResult<i32> {
        match self {
            Err(e) => Err(e),
            Ok(t) => match rank {
                None => Ok(Tensor {
                    shape: vec![],
                    data: vec![t.data.into_iter().sum()],
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
                                acc.iter().zip(chunk.iter()).map(|(a, b)| a + b).collect()
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
                            .map(|chunk| chunk.into_iter().sum())
                            .collect(),
                    })
                }
                Some(_) => Err(TensorError::NotImplementedYet),
            },
        }
    }

    fn maximum(self, rank: Rank) -> TensorResult<i32> {
        match self {
            Err(e) => Err(e),
            Ok(t) => match rank {
                None => Ok(Tensor {
                    shape: vec![],
                    data: vec![t.data.into_iter().max().unwrap()],
                }),
                Some(_) => Err(TensorError::NotImplementedYet),
            },
        }
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
                let n: usize = (*t.shape.iter().next().unwrap()).try_into().unwrap();
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

// pub fn check_matrix(Tensor grid) {
//     grid.eye()
//         .s(rev, id)
//         .equal(grid.min(1))
// }

// pub fn mco(Tensor vector) {
//     vector.scan(phi1(left, mul, plus))
//           .maximum()
// }

// pub fn max_paren_depth(str equation) {
//     equation.to_tensor()
//             .outer("()")
//             .reduce(num::minus, 1)
//             .scan(num::plus)
//             .maximum()
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_sums() {
        {
            // sum all values
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_scalar(45));
            assert_eq!(input.sum(None), expected);
        }
        {
            // sum columns
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_vector(vec![12, 15, 18]));
            assert_eq!(input.sum(Some(1)), expected);
        }
        {
            // sum rows
            let input = Ok(build_scalar(9)).iota().reshape(vec![3, 3]);
            let expected = Ok(build_vector(vec![6, 15, 24]));
            assert_eq!(input.sum(Some(2)), expected);
        }
    }

    #[test]
    fn test_matrix_iota() {
        let iota_matrix = Ok(build_scalar(12)).iota().reshape(vec![3, 4]);
        let expected = Ok(build_matrix(vec![3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
        assert_eq!(iota_matrix, expected);
    }

    #[test]
    fn test_count_negatives() {
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
}
