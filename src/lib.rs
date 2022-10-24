pub fn add(left: usize, right: usize) -> usize {
    left + right
}

// pub fn check_matrix(Tensor grid) {
//     grid.eye()
//         .s(rev, id)
//         .equal(grid.min(1))
// }

// pub fn count_negatives(Tensor matrix) {
//     matrix.lt(0).sum()
// }

// pub fn max_wealth(Tensor accounts) {
//     // what to call max reduce? probably maximum
//     accounts.sum(1).maximum()
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
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

}
