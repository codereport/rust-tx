#![feature(int_roundings)]

use itertools::Itertools;
use iterx::Iterx;
use std::collections::HashSet;
use std::convert::TryInto;
use std::iter;
#[cfg(test)]
use std::ops::Sub;
use std::ops::{Add, Mul};
use std::process::{ExitCode, Termination};

#[derive(Debug, PartialEq)]
pub enum TensorError {
    Domain,
    Rank,
    Shape,
    Type,
    NotImplementedYet,
}

pub type Rank = Option<i32>;

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T> {
    pub shape: Vec<i32>,
    data: Vec<T>,
}

pub type TensorResult<T> = Result<Tensor<T>, TensorError>;

impl<T> Termination for Tensor<T> {
    fn report(self) -> ExitCode {
        ExitCode::SUCCESS
    }
}

impl<T> Tensor<T> {
    pub fn is_empty(&self) -> Tensor<i32> {
        Tensor {
            shape: vec![],
            data: vec![self.data.is_empty().into()],
        }
    }

    pub fn len(&self) -> Tensor<i32> {
        Tensor {
            shape: vec![],
            data: vec![*self.shape.first().unwrap()],
        }
    }

    pub fn rank(&self) -> i32 {
        self.shape.len() as i32
    }

    pub fn to_vec(self) -> Option<Vec<T>> {
        if self.rank() != 1 {
            return None;
        }
        Some(self.data)
    }
}

pub trait TensorOps {
    type Item;

    fn append(self, other: Tensor<Self::Item>) -> TensorResult<Self::Item>;
    fn first(self, rank: Rank) -> TensorResult<Self::Item>;
    fn flatten(self) -> Tensor<Self::Item>;
    fn flatten_ok(self) -> TensorResult<Self::Item>;
    fn chunk(self, chunk_size: usize) -> TensorResult<Self::Item>;
    fn drop_first(self, rank: Rank) -> TensorResult<Self::Item>;
    fn drop_last(self, rank: Rank) -> TensorResult<Self::Item>;
    fn intersection(self, other: Tensor<Self::Item>) -> TensorResult<Self::Item>;
    fn join(self, other: Tensor<Self::Item>) -> TensorResult<Self::Item>;
    fn last(self, rank: Rank) -> TensorResult<Self::Item>;
    fn partition(self, pred: &dyn Fn(&Self::Item) -> bool) -> TensorResult<Self::Item>;
    fn replicate(self, shape: Vec<i32>) -> TensorResult<Self::Item>;
    fn reshape(self, shape: Vec<i32>) -> Tensor<Self::Item>;
    fn reverse(self, rank: Rank) -> TensorResult<Self::Item>;
    fn rotate(self, other: Tensor<i32>, rank: Rank) -> TensorResult<Self::Item>;
    fn slide(self, window_size: usize) -> TensorResult<Self::Item>;
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

    // HOFs
    fn outer_product(
        self,
        other: Tensor<Self::Item>,
        binop: &dyn Fn(Self::Item, Self::Item) -> i32,
    ) -> TensorResult<i32>;
}

pub trait TensorIntOps {
    // Unary Functions
    fn indices(self) -> TensorResult<i32>;
    fn iota(self) -> Tensor<i32>;

    // Unary Scalar Functions
    fn not(self) -> TensorResult<i32>;
    fn sign(self) -> TensorResult<i32>;

    // Binary Functions
    fn base(self, base: i32) -> TensorResult<i32>;

    // Binary Scalar Functions
    fn min(self, other: Tensor<i32>) -> TensorResult<i32>;
    fn multiply(self, other: Tensor<i32>) -> TensorResult<i32>;
    fn plus(self, other: Tensor<i32>) -> TensorResult<i32>;
    fn remainder(self, other: Tensor<i32>) -> TensorResult<i32>;

    // HOFs
    fn reduce<F>(self, binop: F, rank: Rank) -> TensorResult<i32>
    where
        F: Fn(i32, i32) -> i32 + Clone;
    fn scan<F>(self, binop: F, rank: Rank) -> TensorResult<i32>
    where
        F: FnMut(i32, i32) -> i32 + Clone;
    fn prescan<F>(self, init: i32, binop: F, rank: Rank) -> TensorResult<i32>
    where
        F: FnMut(i32, i32) -> i32;
    fn triangle_product(self, binop: &dyn Fn(i32, i32) -> i32) -> TensorResult<i32>;

    // Reduce Specializations
    fn all(self, rank: Rank) -> TensorResult<i32>;
    fn any(self, rank: Rank) -> TensorResult<i32>;
    fn maximum(self, rank: Rank) -> TensorResult<i32>;
    fn minimum(self, rank: Rank) -> TensorResult<i32>;
    fn product(self, rank: Rank) -> TensorResult<i32>;
    fn sum(self, rank: Rank) -> TensorResult<i32>;
}

impl<
        T: std::cmp::PartialOrd<T>
            + std::hash::Hash
            + std::cmp::Eq
            + std::clone::Clone
            + std::marker::Copy
            + std::cmp::Ord
            + std::fmt::Debug,
    > TensorOps for Tensor<T>
{
    type Item = T;

    fn append(self, other: Tensor<Self::Item>) -> TensorResult<Self::Item> {
        if self.rank() == 0 {
            if other.rank() == 2 {
                let value = self.data.first().unwrap();
                let [rows, cols] = other.shape[..] else { return Err(TensorError::Shape) };
                return Ok(Tensor {
                    shape: vec![rows, cols + 1],
                    data: other
                        .data
                        .chunks(cols as usize)
                        .flat_map(|chunk| chunk.iter().copied().prepend(*value))
                        .collect(),
                });
            }
            return Err(TensorError::NotImplementedYet);
        } else if self.rank() == 1 {
            if other.rank() == 2 {
                let [rows, cols] = other.shape[..] else { return Err(TensorError::Shape) };
                if rows != *self.shape.first().unwrap() {
                    return Err(TensorError::Shape);
                }
                return Ok(Tensor {
                    shape: vec![rows, cols + 1],
                    data: other
                        .data
                        .chunks(cols as usize)
                        .zip_map(self.data, |chunk, value| {
                            chunk.iter().copied().prepend(value)
                        })
                        .flatten()
                        .collect(),
                });
            }
            return Err(TensorError::NotImplementedYet);
        }
        Err(TensorError::NotImplementedYet)
    }

    fn first(self, rank: Rank) -> TensorResult<T> {
        if rank.is_none() {
            return Ok(Tensor {
                shape: vec![],
                data: self.data.into_iter().take(1).collect(),
            });
        }
        if rank.unwrap() > self.rank() {
            return Err(TensorError::Rank);
        }
        if self.rank() > 2 {
            return Err(TensorError::NotImplementedYet);
        }
        match rank {
            Some(2) => {
                let new_shape = vec![*self.shape.first().unwrap()];
                let chunk_size = *self.shape.last().unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .flat_map(|chunk| chunk.iter().copied().take(1))
                        .collect(),
                })
            }
            _ => Err(TensorError::NotImplementedYet),
        }
    }

    fn drop_first(self, rank: Rank) -> TensorResult<T> {
        match (self.rank(), rank) {
            (1, None) => Ok(Tensor {
                shape: vec![*self.shape.first().unwrap() - 1],
                data: self.data.into_iter().skip(1).collect(),
            }),
            (2, Some(2)) => {
                let [rows, cols] = self.shape[..] else { return Err(TensorError::Shape) };
                Ok(Tensor {
                    shape: vec![rows, cols - 1],
                    data: self
                        .data
                        .chunks(cols as usize)
                        .flat_map(|chunk| chunk.iter().copied().skip(1))
                        .collect(),
                })
            }
            _ => Err(TensorError::NotImplementedYet),
        }
    }

    fn drop_last(self, rank: Rank) -> TensorResult<T> {
        match (self.rank(), rank) {
            (1, None) => Ok(Tensor {
                shape: vec![*self.shape.first().unwrap() - 1],
                data: self.data.into_iter().drop_last().collect(),
            }),
            (2, Some(2)) => {
                let [rows, cols] = self.shape[..] else { return Err(TensorError::Shape) };
                Ok(Tensor {
                    shape: vec![rows, cols - 1],
                    data: self
                        .data
                        .chunks(cols as usize)
                        .flat_map(|chunk| chunk.iter().copied().drop_last())
                        .collect(),
                })
            }
            _ => Err(TensorError::NotImplementedYet),
        }
    }

    fn last(self, rank: Rank) -> TensorResult<T> {
        if rank.is_none() {
            return Ok(Tensor {
                shape: vec![],
                data: vec![*self.data.last().unwrap()],
            });
        }
        if rank.unwrap() > self.rank() {
            return Err(TensorError::Rank);
        }
        if self.rank() > 2 {
            return Err(TensorError::NotImplementedYet);
        }
        match rank {
            Some(2) => {
                let new_shape = vec![*self.shape.first().unwrap()];
                let chunk_size = *self.shape.last().unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .flat_map(|chunk| chunk.iter().copied().skip(chunk_size - 1))
                        .collect(),
                })
            }
            _ => Err(TensorError::NotImplementedYet),
        }
    }

    /// Returns the intersection (common elements) of two rank-1 `Tensor`s.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rust_tx::*;
    /// # use std::io;
    /// #
    /// # fn main() -> TensorResult<i32> {
    /// let a = build_scalar(3).iota();                        // 1 2 3
    /// let b = build_scalar(3).iota().plus(build_scalar(2))?; // 3 4 5
    /// # // let c = build_scalar(3).iota().plus(3); // 4 5 6
    ///
    /// assert_eq!(a.intersection(b)?, build_vector(vec![3]));
    /// # Ok(build_scalar(1))
    /// # }
    /// ```
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
                    .zip(other.data)
                    .map(|(a, b)| binop(a, b))
                    .collect(),
            });
        }
        if self.rank() == 2 && other.rank() == 1 {
            if *self.shape.first().unwrap() == *other.shape.first().unwrap() {
                let x = *self.shape.first().unwrap() as usize;
                let n = *self.shape.last().unwrap();
                return Ok(Tensor {
                    shape: self.shape,
                    data: self
                        .data
                        .into_iter()
                        .zip(other.replicate(vec![n; x])?.to_vec().unwrap())
                        .map(|(a, b)| binop(a, b))
                        .collect(),
                });
            }
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

    fn join(self, other: Tensor<T>) -> TensorResult<T> {
        if self.rank() > 1 || other.rank() > 1 {
            return Err(TensorError::NotImplementedYet);
        }
        Ok(Tensor {
            shape: vec![(self.data.len() + other.data.len()).try_into().unwrap()],
            data: self.data.into_iter().chain(other.data).collect(),
        })
    }

    fn outer_product(self, other: Tensor<T>, binop: &dyn Fn(T, T) -> i32) -> TensorResult<i32> {
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

    fn partition(self, pred: &dyn Fn(&Self::Item) -> bool) -> TensorResult<Self::Item> {
        if self.rank() != 1 {
            return Err(TensorError::NotImplementedYet);
        }
        let (front, back): (Vec<_>, Vec<_>) = self.data.into_iter().partition(pred);
        Ok(Tensor {
            shape: self.shape,
            data: front.into_iter().chain(back).collect(),
        })
    }

    fn flatten(self) -> Tensor<T> {
        Tensor {
            shape: vec![self.shape.iter().product()],
            data: self.data,
        }
    }

    fn flatten_ok(self) -> TensorResult<T> {
        Ok(self.flatten())
    }

    fn replicate(self, amounts: Vec<i32>) -> TensorResult<T> {
        if self.rank() != 1 {
            return Err(TensorError::NotImplementedYet);
        }
        if build_vector(amounts.clone()).len() != self.len() {
            return Err(TensorError::Shape);
        }
        Ok(Tensor {
            shape: vec![amounts.iter().sum()],
            data: self
                .data
                .into_iter()
                .zip(amounts)
                .flat_map(|(value, amount)| std::iter::repeat(value).take(amount as usize))
                .collect(),
        })
    }

    fn reshape(self, shape: Vec<i32>) -> Tensor<T> {
        let n: i32 = shape.iter().product();
        Tensor {
            shape,
            data: self.data.into_iter().cycle().take(n as usize).collect(),
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

    fn rotate(self, other: Tensor<i32>, rank: Rank) -> TensorResult<T> {
        if other.rank() > 0 {
            return Err(TensorError::Rank);
        }
        let n: i32 = *other.data.first().unwrap();
        let l: i32 = self.data.len().try_into().unwrap();
        let nonneg_n: usize = (if n < 0 { l + n } else { n }).try_into().unwrap();
        match rank {
            Some(_) => Err(TensorError::NotImplementedYet),
            None => Ok(Tensor {
                shape: self.shape,
                data: self
                    .data
                    .clone()
                    .into_iter()
                    .cycle()
                    .skip(nonneg_n)
                    .take(self.data.len())
                    .collect(),
            }),
        }
    }

    fn chunk(self, chunk_size: usize) -> TensorResult<T> {
        if self.rank() != 1 {
            return Err(TensorError::Rank);
        } else if self.data.len() % chunk_size != 0 {
            return Err(TensorError::Shape);
        }

        Ok(Tensor {
            shape: vec![(self.data.len() / chunk_size) as i32, chunk_size as i32],
            data: self.data,
        })
    }

    fn slide(self, window_size: usize) -> TensorResult<T> {
        if self.rank() != 1 {
            return Err(TensorError::Rank);
        }
        Ok(Tensor {
            shape: vec![
                (self.data.len() - window_size + 1) as i32,
                window_size as i32,
            ],
            data: self
                .data
                .windows(window_size)
                .flat_map(|x| x.iter().copied())
                .collect::<Vec<_>>(),
        })
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

fn domain_check(op: &dyn Fn(i32) -> i32) -> impl Fn(i32) -> Result<i32, TensorError> + '_ {
    |x: i32| {
        if !(0..=1).contains(&x) {
            Err(TensorError::Domain)
        } else {
            Ok(op(x))
        }
    }
}

impl TensorIntOps for Tensor<i32> {
    fn indices(self) -> TensorResult<i32> {
        match self.rank() {
            1 => {
                let new_data = self
                    .data
                    .into_iter()
                    .enumerate()
                    .filter(|&(_, x)| x == 1)
                    .map(|(i, _)| i as i32 + 1)
                    .collect::<Vec<_>>();
                Ok(Tensor {
                    shape: vec![new_data.len() as i32],
                    data: new_data,
                })
            }
            2 => {
                let [rows, cols] = self.shape[..] else { return Err(TensorError::Shape) };
                let new_data = (1..=rows)
                    .cartesian_product(1..=cols)
                    .zip(self.data)
                    .filter(|&(_, x)| x == 1)
                    .flat_map(|((i, j), _)| [i, j])
                    .collect::<Vec<_>>();
                Ok(Tensor {
                    shape: vec![new_data.len() as i32 / 2, 2],
                    data: new_data,
                })
            }
            _ => Err(TensorError::NotImplementedYet),
        }
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

    fn plus(self, other: Tensor<i32>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &Add::add)
    }

    fn remainder(self, other: Tensor<i32>) -> TensorResult<i32> {
        self.scalar_binary_operation(other, &i32::rem_euclid)
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

    fn reduce<F>(self, binop: F, rank: Rank) -> TensorResult<i32>
    where
        F: Fn(i32, i32) -> i32 + Clone,
    {
        match rank {
            None => Ok(Tensor {
                shape: vec![],
                data: vec![self.data.into_iter().reduce(binop).unwrap()],
            }),
            Some(1) => {
                // TODO: only works for matrices
                let new_shape: Vec<i32> = self.shape.clone().into_iter().skip(1).collect();
                let chunk_size = *self.shape.last().unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .fold(vec![0; chunk_size], |acc, chunk| {
                            acc.into_iter()
                                .zip(chunk.iter().copied())
                                .map(|(a, b)| binop(a, b))
                                .collect()
                            // TODO: get zip_map to work here :/
                            // acc.into_iter()
                            //     .zip_map(chunk.iter().copied(), binop)
                            //     .collect()
                        }),
                })
            }
            Some(2) => {
                // TODO: only works for matrices
                let new_shape: Vec<i32> = self.shape.clone().into_iter().take(1).collect();
                let chunk_size = *self.shape.last().unwrap() as usize;
                Ok(Tensor {
                    shape: new_shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .map(|chunk| chunk.iter().copied().reduce(binop.clone()).unwrap())
                        .collect(),
                })
            }
            Some(_) => Err(TensorError::NotImplementedYet),
        }
    }

    fn scan<F>(self, binop: F, rank: Rank) -> TensorResult<i32>
    where
        F: FnMut(i32, i32) -> i32 + Clone,
    {
        match rank {
            None => Ok(Tensor {
                shape: self.shape,
                data: self.data.into_iter().scan_(binop).collect(),
            }),
            Some(2) => {
                let chunk_size = *self.shape.last().unwrap() as usize;
                Ok(Tensor {
                    shape: self.shape,
                    data: self
                        .data
                        .chunks(chunk_size)
                        .flat_map(|chunk| chunk.iter().copied().scan_(binop.clone()))
                        .collect(),
                })
            }
            Some(_) => Err(TensorError::NotImplementedYet),
        }
    }

    fn prescan<F>(self, init: i32, binop: F, rank: Rank) -> TensorResult<i32>
    where
        F: FnMut(i32, i32) -> i32,
    {
        match rank {
            None => Ok(Tensor {
                shape: vec![*self.shape.first().unwrap() + 1],
                data: self.data.into_iter().prescan(init, binop).collect(),
            }),
            Some(_) => Err(TensorError::NotImplementedYet),
        }
    }

    fn not(self) -> TensorResult<i32> {
        Ok(Tensor {
            shape: self.shape,
            data: self
                .data
                .into_iter()
                .map(domain_check(&|x| (x == 0).into()))
                .collect::<Result<_, _>>()?,
        })
    }

    fn sign(self) -> TensorResult<i32> {
        Ok(Tensor {
            shape: self.shape,
            data: self.data.into_iter().map(num::signum).collect(),
        })
    }

    fn base(self, base: i32) -> TensorResult<i32> {
        if self.rank() > 1 {
            return Err(TensorError::NotImplementedYet);
        }
        let base_k = |v: i32, b: i32, n: usize| {
            iter::repeat(b)
                .take(n)
                .fold((vec![], v), |(v, t), x| {
                    ([vec![t.rem_euclid(x)], v].concat(), t.div_floor(x))
                })
                .0
        };
        if self.rank() == 0 {
            let val: i32 = *self.data.first().unwrap();
            let n: usize = (val.ilog(base) + 1).try_into().unwrap();
            Ok(Tensor {
                shape: vec![n as i32],
                data: base_k(val, base, n),
            })
        } else {
            let val: i32 = *self.clone().maximum(None)?.data.first().unwrap();
            let n: usize = (val.ilog(base) + 1).try_into().unwrap();
            Ok(Tensor {
                shape: [self.shape, vec![n as i32]].concat(),
                data: self
                    .data
                    .into_iter()
                    .flat_map(|x| base_k(x, base, n))
                    .collect(),
            })
        }
    }

    fn all(self, rank: Rank) -> TensorResult<i32> {
        self.reduce(&std::cmp::min, rank)?.min(build_scalar(1))
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

pub fn build_vector_from_string(str: String) -> Tensor<char> {
    build_vector(str.chars().collect::<Vec<_>>())
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

#[cfg(test)]
fn count_negatives(nums: Tensor<i32>) -> TensorResult<i32> {
    let n = build_scalar(0);
    nums.less_than(n)?.sum(None)
}

// pub fn count_negatives(nums: Tensor) {
//     nums.lt(0).sum()
// }

#[cfg(test)]
fn max_wealth(accounts: Tensor<i32>) -> TensorResult<i32> {
    accounts.sum(Some(2))?.maximum(None)
}

// pub fn max_wealth(accounts: Tensor) {
//     accounts.sum(1).maximum()
// }

#[cfg(test)]
fn array_sign(arr: Tensor<i32>) -> TensorResult<i32> {
    arr.sign()?.product(None)
}

// pub fn array_sign(arr: Tensor<i32>) {
//     arr.sign().product()
// }

#[cfg(test)]
fn mco(vec: Tensor<i32>) -> TensorResult<i32> {
    vec.scan(|a, b| b * (a + b), None)?.maximum(None)
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

#[cfg(test)]
fn stringless_max_paren_depth(equation: Tensor<i32>) -> TensorResult<i32> {
    let rhs = build_vector(vec![2, 3]);
    equation
        .outer_product(rhs, &|a, b| (a == b).into())?
        .reduce(&Sub::sub, Some(2))?
        .scan(|x, y| x + y, None)?
        .maximum(None)
}

#[cfg(test)]
fn smaller_numbers_than_current(nums: Tensor<i32>) -> TensorResult<i32> {
    nums.clone()
        .outer_product(nums, &|a, b| (a > b).into())?
        .sum(Some(2))
}

#[cfg(test)]
fn find_pairs(nums: Tensor<i32>, k: i32) -> TensorResult<i32> {
    let uniq = nums.unique()?;
    uniq.clone()
        .outer_product(uniq, &Sub::sub)?
        .equal(build_scalar(k))?
        .sum(None)
}

#[cfg(test)]
fn lucky_numbers(matrix: Tensor<i32>) -> TensorResult<i32> {
    matrix
        .clone()
        .minimum(Some(2))?
        .intersection(matrix.maximum(Some(1))?)
}

#[cfg(test)]
fn num_special(mat: Tensor<i32>) -> TensorResult<i32> {
    mat.clone()
        .sum(Some(1))?
        .multiply(mat.sum(Some(2))?)?
        .equal(build_scalar(1))?
        .sum(None)
}

#[cfg(test)]
fn num_identical_pairs(nums: Tensor<i32>) -> TensorResult<i32> {
    nums.triangle_product(&|a, b| (a == b).into())?.sum(None)
}

#[cfg(test)]
fn max_ice_cream(costs: Tensor<i32>, coins: Tensor<i32>) -> TensorResult<i32> {
    costs
        .sort()?
        .scan(|x, y| x + y, None)?
        .less_than_or_equal(coins)?
        .sum(None)
}

#[cfg(test)]
fn can_make_arithmetic_progression(arr: Tensor<i32>) -> TensorResult<i32> {
    arr.sort()?
        .slide(2)?
        .reduce(&Sub::sub, Some(2))?
        .unique()?
        .len()
        .equal(build_scalar(1))
}

#[cfg(test)]
fn first_uniq_num(nums: Tensor<i32>) -> TensorResult<i32> {
    nums.clone()
        .outer_product(nums, &|a, b| (a == b).into())?
        .sum(Some(2))?
        .equal(build_scalar(1))?
        .indices()?
        .first(None)
}

#[cfg(test)]
fn check_if_double_exists(arr: Tensor<i32>) -> TensorResult<i32> {
    arr.triangle_product(&|a, b| (a == 2 * b).into())?.any(None)
}

#[cfg(test)]
fn has_alternating_bits(n: Tensor<i32>) -> TensorResult<i32> {
    n.base(2)?
        .slide(2)?
        .reduce(&|a, b| (a != b).into(), Some(2))?
        .all(None)
}

#[cfg(test)]
fn sum_digits_in_base_k(n: Tensor<i32>, k: i32) -> TensorResult<i32> {
    n.base(k)?.sum(None)
}

#[cfg(test)]
fn count_even_digit_sum(num: Tensor<i32>) -> TensorResult<i32> {
    num.iota()
        .base(10)?
        .sum(Some(2))?
        .remainder(build_scalar(2))?
        .not()?
        .sum(None)
}

#[cfg(test)]
fn apply_array_operations(nums: Tensor<i32>) -> TensorResult<i32> {
    let mask = nums
        .clone()
        .slide(2)?
        .reduce(&|a, b| (a == b).into(), Some(2))?
        .plus(build_scalar(1))?;
    mask.clone()
        .rotate(build_scalar(-1), None)?
        .equal(build_scalar(2))?
        .not()?
        .multiply(mask)?
        .join(build_scalar(1))?
        .multiply(nums)?
        .partition(&|x| *x != 0)
}

#[cfg(test)]
fn check_if_pangram(sentence: Tensor<char>) -> TensorResult<i32> {
    sentence.unique()?.len().equal(build_scalar(26))
}

#[cfg(test)]
fn max_length_between_equal_characters(s: Tensor<char>) -> TensorResult<i32> {
    s.clone()
        .outer_product(s, &|a, b| (a == b).into())?
        .indices()?
        .reduce(&Sub::sub, Some(2))?
        .maximum(None)?
        .plus(build_scalar(-1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append() {
        // {
        //     // not supported yet
        //     let first = build_scalar(0);
        //     let rest = build_vector(vec![1, 2, 3]);
        //     let expected = build_vector(vec![0, 1, 2, 3]);
        //     assert_eq!(first.append(rest).unwrap(), expected);
        // }

        {
            let first = build_scalar(0);
            let rest = build_matrix(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
            let expected = build_matrix(vec![2, 4], vec![0, 1, 2, 3, 0, 4, 5, 6]);
            assert_eq!(first.append(rest).unwrap(), expected);
        }

        {
            let first = build_vector(vec![0, 0]);
            let rest = build_matrix(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
            let expected = build_matrix(vec![2, 4], vec![0, 1, 2, 3, 0, 4, 5, 6]);
            assert_eq!(first.append(rest).unwrap(), expected);
        }
    }

    #[test]
    fn test_first() {
        {
            let input = build_vector(vec![1, 2, 3]);
            let expected = build_scalar(1);
            assert_eq!(input.first(None).unwrap(), expected);
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
    fn test_chunk() {
        {
            let input = build_scalar(6).iota();
            let expected = build_matrix(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
            assert_eq!(input.chunk(3).unwrap(), expected);
        }
        {
            let input = build_scalar(6).iota();
            let expected = build_matrix(vec![3, 2], vec![1, 2, 3, 4, 5, 6]);
            assert_eq!(input.chunk(2).unwrap(), expected);
        }
    }

    #[test]
    fn test_reshape() {
        {
            let input = build_scalar(3).iota();
            let expected = build_matrix(vec![2, 3], vec![1, 2, 3, 1, 2, 3]);
            assert_eq!(input.reshape(vec![2, 3]), expected);
        }
    }

    #[test]
    fn test_replicate() {
        {
            let input = build_scalar(4).iota();
            let expected = build_vector(vec![1, 2, 2, 3, 3, 3]);
            assert_eq!(input.replicate(vec![1, 2, 3, 0]).unwrap(), expected);
        }
    }

    #[test]
    fn test_plus() {
        {
            let a = build_matrix(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
            let b = build_vector(vec![1, 10]);
            let expected = build_matrix(vec![2, 3], vec![2, 3, 4, 14, 15, 16]);
            assert_eq!(a.plus(b).unwrap(), expected);
        }
    }

    #[test]
    fn test_flatten() {
        {
            let input = build_matrix(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
            let expected = build_scalar(6).iota();
            assert_eq!(input.flatten(), expected);
        }
        {
            let input = build_matrix(vec![3, 2], vec![1, 2, 3, 4, 5, 6]);
            let expected = build_scalar(6).iota();
            assert_eq!(input.flatten(), expected);
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
    fn test_base() {
        assert_eq!(
            build_scalar(4).base(2).unwrap(),
            build_vector(vec![1, 0, 0])
        );

        assert_eq!(
            build_scalar(123).base(10).unwrap(),
            build_vector(vec![1, 2, 3])
        );
    }

    #[test]
    fn test_not() {
        assert_eq!(
            build_vector(vec![1, 0, 1]).not().unwrap(),
            build_vector(vec![0, 1, 0])
        );

        assert_eq!(build_vector(vec![1, 2, 3]).not(), Err(TensorError::Domain));
    }

    #[test]
    fn test_len_is_empty() {
        assert_eq!(build_vector(vec![1, 2, 3]).len(), build_scalar(3));
        assert_eq!(build_vector(vec![1, 2, 3]).is_empty(), build_scalar(0));
        assert_eq!(build_vector::<Vec<i32>>(vec![]).is_empty(), build_scalar(1));
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

    #[test]
    fn test_has_alternating_bits() {
        // https://leetcode.com/problems/binary-number-with-alternating-bits/
        {
            let input = build_scalar(5);
            let expected = build_scalar(1);
            assert_eq!(has_alternating_bits(input).unwrap(), expected);
        }
        {
            let input = build_scalar(7);
            let expected = build_scalar(0);
            assert_eq!(has_alternating_bits(input).unwrap(), expected);
        }
        {
            let input = build_scalar(11);
            let expected = build_scalar(0);
            assert_eq!(has_alternating_bits(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_sum_digits_in_base_k() {
        // https://leetcode.com/problems/sum-of-digits-in-base-k/
        {
            let input = build_scalar(34);
            let expected = build_scalar(9);
            assert_eq!(sum_digits_in_base_k(input, 6).unwrap(), expected);
        }
        {
            let input = build_scalar(10);
            let expected = build_scalar(1);
            assert_eq!(sum_digits_in_base_k(input, 10).unwrap(), expected);
        }
    }

    #[test]
    fn test_count_even_digit_sum() {
        // https://leetcode.com/problems/count-integers-with-even-digit-sum/
        {
            let input = build_scalar(4);
            let expected = build_scalar(2);
            assert_eq!(count_even_digit_sum(input).unwrap(), expected);
        }
        {
            let input = build_scalar(30);
            let expected = build_scalar(14);
            assert_eq!(count_even_digit_sum(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_apply_array_operations() {
        // https://leetcode.com/problems/apply-operations-to-an-array/
        {
            let input = build_vector(vec![1, 2, 2, 1, 1, 0]);
            let expected = build_vector(vec![1, 4, 2, 0, 0, 0]);
            assert_eq!(apply_array_operations(input).unwrap(), expected);
        }
        {
            let input = build_vector(vec![0, 1]);
            let expected = build_vector(vec![1, 0]);
            assert_eq!(apply_array_operations(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_check_if_pangram() {
        {
            let input = build_vector_from_string("thequickbrownfoxjumpsoverthelazydog".to_string());
            let expected = build_scalar(1);
            assert_eq!(check_if_pangram(input).unwrap(), expected);
        }
        {
            let input = build_vector_from_string("leetcode".to_string());
            let expected = build_scalar(0);
            assert_eq!(check_if_pangram(input).unwrap(), expected);
        }
    }

    #[test]
    fn test_max_length_between_equal_characters() {
        {
            let input = build_vector_from_string("aa".to_string());
            let expected = build_scalar(0);
            assert_eq!(
                max_length_between_equal_characters(input).unwrap(),
                expected
            );
        }
        {
            let input = build_vector_from_string("abca".to_string());
            let expected = build_scalar(2);
            assert_eq!(
                max_length_between_equal_characters(input).unwrap(),
                expected
            );
        }
        {
            let input = build_vector_from_string("cbzxy".to_string());
            let expected = build_scalar(-1);
            assert_eq!(
                max_length_between_equal_characters(input).unwrap(),
                expected
            );
        }
    }

    // Leetcode Problems TODO:
    // https://leetcode.com/problems/three-consecutive-odds/
    // tco ← {3≤⌈/≢¨⊆⍨2|⍵}  ⍝ shorter solution
    // https://leetcode.com/problems/maximum-ascending-subarray-sum/
    // maxAscendingSum ← ⌈/(+/¨((1,2>/⊢)⊂⊢))
    // https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/
    // makeEqual s = all (==0) . map (\x -> mod x (length s)) . map snd . count $ concat s
}
